"""Tiny end-to-end shape/wiring check. Run this before submitting a job.

    uv run src/smoke_test.py

Builds a miniature config and pushes synthetic data through every path the
training and evaluation scripts use: encode, teacher-forced step, multi-step
rollout, dream, actor, autoregressive actor, backtest. It asserts shapes and
that losses are finite -- it does not check that anything learns.
"""

import copy

import torch

from actor import Actor, ActorAR
from jepa import JEPA
from util import (bin_centers, delta_equity, equity, latent_diagnostics,
                  loss_fn_so, make_constrained_loss, optimal_allocation,
                  preprocess_classes)

B, S, W, F = 4, 8, 5, 6
RET_BINS, ACT_BINS = 11, 9

CFG = {
    'experiment_name': 'smoke',
    'data': {
        'window_size': W, 'sequence_length': S, 'stride': S, 'eval_stride': S,
        'normalization_window': 100, 'vol_window_short': 10,
        'returns': {'num_bins': RET_BINS, 'min_value': -5.0, 'max_value': 5.0},
        'actions': {'num_bins': ACT_BINS, 'min_value': -1.0, 'max_value': 1.0,
                    'commission_value': 0.0005, 'max_change': 0.1,
                    'penalty_weight': 1000.0, 'opt_steps': 20, 'opt_lr': 0.05},
    },
    'jepa': {
        'name': 'jepa', 'input_dim': F, 'd_model': 32,
        'lam_state': 1.0, 'lam_CE': 1.0, 'lam_SIGReg': 0.1, 'lam_rollout': 0.5,
        'sigreg': {'knots': 17, 'num_proj': 64},
        'encoder': {'num_layers': 2, 'num_heads': 4, 'ff_mult': 4, 'max_len': W + 1, 'dropout': 0.1},
        'predictor': {'num_layers': 2, 'num_heads': 4, 'ff_mult': 4, 'max_len': S, 'dropout': 0.1},
        'return_head': {'num_bins': RET_BINS, 'min_value': -5.0, 'max_value': 5.0, 'dropout': 0.1},
        'training': {'lr': 3e-4, 'weight_decay': 0.01, 'warmup_steps': 10,
                     'sched_steps': 100, 'epochs': 1, 'batch_size': B,
                     'rollout_steps': 4, 'rollout_batch': 2,
                     'log_every_n_steps': 1, 'patience': 1},
    },
    'actor': {
        'name': 'actor', 'input_dim': 32, 'd_model': 32,
        'backbone': {'num_layers': 2, 'num_heads': 4, 'ff_mult': 4, 'max_len': S, 'dropout': 0.1},
        'action_head': {'num_bins': ACT_BINS, 'min_value': -1.0, 'max_value': 1.0, 'dropout': 0.1},
        'ar': {'ctx_len': 2, 'pred_steps': 5},
        'eval': {'ctx_len': 1, 'pred_steps': None, 'val_pred_steps': 4},
        'training': {'lr': 3e-4, 'weight_decay': 0.01, 'warmup_steps': 10,
                     'sched_steps': 100, 'epochs': 1, 'batch_size': B,
                     'log_every_n_steps': 1, 'patience': 1},
        'test': {'batch_size': B, 'act_temp': 1.0, 'num_plots': 1},
    },
}

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f'  ok    {name}')
    except Exception as exc:
        FAIL.append((name, exc))
        print(f'  FAIL  {name}: {type(exc).__name__}: {exc}')


def eq(actual, expected, what):
    assert tuple(actual) == tuple(expected), f'{what}: got {tuple(actual)}, want {tuple(expected)}'


def finite(t, what):
    assert torch.isfinite(t).all(), f'{what} is not finite: {t}'


def make_batch():
    g = torch.Generator().manual_seed(0)
    ret_norm = torch.randn(B, S, generator=g)
    ret, ret_target = preprocess_classes(ret_norm, -5.0, 5.0, RET_BINS)
    act_raw = torch.tanh(torch.randn(B, S, generator=g))
    act, act_target = preprocess_classes(act_raw, -1.0, 1.0, ACT_BINS)
    return {
        'sample': torch.randn(B, S, W, F, generator=g),
        'return': ret,
        'return_target': ret_target,
        'return_raw': (ret_norm * 0.004).unsqueeze(-1),
        'vol': torch.full((B, S, 1), 0.004),
        'action': act,
        'action_target': act_target,
    }


def main():
    torch.manual_seed(0)
    batch = make_batch()

    print('\nutil')

    def t_bins():
        c = bin_centers(-1.0, 1.0, 4)
        eq(c.shape, (4,), 'bin_centers')
        # Centres, not edges: for 4 bins over [-1, 1] they are -0.75..0.75.
        assert torch.allclose(c, torch.tensor([-0.75, -0.25, 0.25, 0.75])), c
    check('bin_centers are centres', t_bins)

    def t_classes():
        v, i = preprocess_classes(torch.tensor([[-9.0, 0.0, 9.0]]), -5.0, 5.0, RET_BINS)
        eq(v.shape, (1, 3, 1), 'clipped values')
        eq(i.shape, (1, 3, 1), 'bin indices')
        assert v.min() >= -5.0 and v.max() <= 5.0, 'values must be clipped'
        assert i.min() >= 0 and i.max() < RET_BINS, 'indices out of range'
    check('preprocess_classes clips and bins', t_classes)

    def t_sortino():
        # A path that only ever loses must not score well. The old formulation
        # measured dispersion of losses, so a uniformly-losing path scored +inf.
        losing = torch.full((1, 5), -0.01)
        always_long = torch.full((1, 6), 1.0)
        bad = loss_fn_so(always_long, losing, 0.0005)
        winning = torch.full((1, 5), 0.01)
        good = loss_fn_so(always_long, winning, 0.0005)
        assert good < bad, f'winning path must score better: good={good}, bad={bad}'
        finite(bad, 'sortino(losing)')
    check('loss_fn_so ranks winning above losing', t_sortino)

    def t_equity():
        x = torch.zeros(2, 6)
        E, e = equity(x, torch.randn(2, 5) * 0.01, 0.0005)
        eq(E.shape, (2, 5), 'equity curve')
        # Flat allocation trades nothing and earns nothing.
        assert torch.allclose(e, torch.ones(2)), e
    check('equity of a flat book is 1.0', t_equity)

    def t_alloc():
        p = torch.randn(3, 6) * 0.01
        a = optimal_allocation(p, 0.0005, loss_fn=make_constrained_loss(loss_fn_so),
                               lr=0.05, steps=20)
        eq(a.shape, (3, 7), 'oracle path (leading x0 included)')
        assert a.abs().max() <= 1.0, 'allocation must stay in [-1, 1]'
        assert not a.requires_grad, 'oracle labels must be detached'
    check('optimal_allocation shape and range', t_alloc)

    print('\njepa')
    jepa = JEPA(copy.deepcopy(CFG))
    jepa.train()

    def t_encode():
        Z = jepa.encode(batch['sample'])
        eq(Z.shape, (B, S, 32), 'latents')
        finite(Z, 'latents')
    check('encode', t_encode)

    def t_predict():
        Z = jepa.encode(batch['sample'])
        Zp1, logits = jepa.predict(Z[:, :-1], batch['return'][:, :-1])
        eq(Zp1.shape, (B, S - 1, 32), 'predicted latents')
        eq(logits.shape, (B, S - 1, RET_BINS), 'return logits')
    check('predict', t_predict)

    def t_sigreg():
        Z = jepa.encode(batch['sample'])
        s = jepa.sigreg(Z)
        eq(s.shape, (), 'sigreg is a scalar')
        finite(s, 'sigreg')
        assert s > 0, 'sigreg on random latents should be positive'
    check('SIGReg', t_sigreg)

    def t_diag():
        d = latent_diagnostics(torch.randn(B, S, 32))
        assert d['latent_erank'] > 1.0, 'random latents should have rank > 1'
        # A constant (fully collapsed) latent must be detected.
        c = latent_diagnostics(torch.ones(B, S, 32))
        assert c['latent_erank'] < 1.5, f"collapse not detected: {c['latent_erank']}"
    check('latent_diagnostics detects collapse', t_diag)

    def t_rollout():
        Z = jepa.encode(batch['sample'])
        pz, pl = jepa.rollout(Z[:, :2], batch['return'][:, :2], 4)
        eq(pz.shape, (B, 4, 32), 'rollout latents')
        eq(pl.shape, (B, 4, RET_BINS), 'rollout logits')
        assert pz.requires_grad, 'rollout must stay differentiable'
    check('multi-step rollout', t_rollout)

    def t_step():
        loss = jepa.training_step(batch, 0)
        eq(loss.shape, (), 'loss is a scalar')
        finite(loss, 'jepa loss')
        loss.backward()
        grads = [p.grad for p in jepa.parameters() if p.grad is not None]
        assert grads, 'no gradients reached the parameters'
        assert all(torch.isfinite(g).all() for g in grads), 'non-finite gradient'
    check('training_step + backward', t_step)

    def t_dream():
        jepa.eval()
        with torch.no_grad():
            Z = jepa.encode(batch['sample'])
            ds, dr, dp, di = jepa.dream(Z[:, :3], batch['return'][:, :3], horizon=4)
        eq(ds.shape, (B, 4, 32), 'dream latents')
        eq(dr.shape, (B, 4, 1), 'dream returns')
        eq(dp.shape, (B, 4, RET_BINS), 'dream probabilities')
        eq(di.shape, (B, 4, 1), 'dream indices')
        jepa.train()
    check('dream', t_dream)

    def t_truncate():
        # Longer than predictor.max_len: x and cond must be truncated together.
        jepa.eval()
        with torch.no_grad():
            long_Z = torch.randn(B, S + 6, 32)
            long_R = torch.randn(B, S + 6, 1)
            out, lg = jepa.predict(long_Z, long_R)
        eq(out.shape, (B, S, 32), 'truncated predictor output')
        eq(lg.shape, (B, S, RET_BINS), 'truncated logits')
        jepa.train()
    check('predictor truncation past max_len', t_truncate)

    print('\nactor')
    jepa.eval()
    actor = Actor(copy.deepcopy(CFG), jepa)
    actor.train()

    def t_cond_dim():
        assert actor.condition_dim == RET_BINS + 1, actor.condition_dim
    check('condition = return bins + previous allocation', t_cond_dim)

    def t_actor_step():
        # The JEPA-only test above already populated .grad on the world model;
        # clear it so this test measures leakage from the ACTOR's backward.
        for p in actor.jepa.parameters():
            p.grad = None
        loss = actor.training_step(batch, 0)
        finite(loss, 'actor loss')
        loss.backward()
        assert all(not p.requires_grad for p in actor.jepa.parameters()), 'world model must stay frozen'
        assert actor.jepa.encoder.cls_token.grad is None, 'gradient leaked into the world model'
    check('training_step keeps the world model frozen', t_actor_step)

    def t_backtest():
        actor.eval()
        with torch.no_grad():
            b = actor.backtest(batch)
        # eval ctx_len defaults to 1, so the rollout covers windows 1..S-1 --
        # the same span the (leaky) teacher-forced pass used to report.
        eq(b['action'].shape, (B, S - 1, 1), 'actions')
        eq(b['equity'].shape, (B, S - 1), 'equity curve')
        eq(b['end_equity'].shape, (B,), 'end equity')
        eq(b['return_prob'].shape, (B, S - 1, RET_BINS), 'return probabilities')
        for k in ('end_equity', 'opt_end_equity', 'bh_end_equity'):
            finite(b[k], k)
        assert b['action'].abs().max() <= 1.0, 'allocation out of range'
        actor.train()
    check('backtest with oracle and buy-and-hold baselines', t_backtest)

    def t_backtest_no_oracle_leak():
        # The default backtest must not read the oracle allocations at all --
        # they are fitted over the whole future. Corrupting them must leave the
        # policy's own equity untouched (only the oracle baseline may move).
        actor.eval()
        dirty = dict(batch)
        dirty['action'] = -batch['action']
        with torch.no_grad():
            b = actor.backtest(batch)
            d = actor.backtest(dirty)
            tf = actor.backtest(batch, teacher_force=True)
            tf_d = actor.backtest(dirty, teacher_force=True)
        assert torch.allclose(b['action'], d['action']), (
            'autoregressive backtest is reading the oracle action path')
        assert not torch.allclose(tf['action'], tf_d['action']), (
            'teacher-forced pass should depend on the oracle -- that is the point of it')
        eq(tf['action'].shape, (B, S - 1, 1), 'teacher-forced actions')
        actor.train()
    check('backtest does not condition on oracle actions', t_backtest_no_oracle_leak)

    def t_backtest_ctx():
        actor.eval()
        with torch.no_grad():
            b = actor.backtest(batch, ctx_len=3, pred_steps=4)
        eq(b['action'].shape, (B, 4, 1), 'short-horizon actions')
        eq(b['return_prob'].shape, (B, 4, RET_BINS), 'short-horizon return probabilities')
        actor.train()
    check('backtest honours ctx_len / pred_steps', t_backtest_ctx)

    def t_oracle_beats():
        actor.eval()
        with torch.no_grad():
            b = actor.backtest(batch)
        # The oracle optimises this exact path, so it must not lose to an
        # untrained policy. A failure here means the labels are misaligned.
        assert b['opt_end_equity'].mean() >= b['end_equity'].mean(), (
            f"oracle {b['opt_end_equity'].mean():.4f} < actor {b['end_equity'].mean():.4f}")
        actor.train()
    check('oracle beats the untrained actor', t_oracle_beats)

    print('\nactor-ar')
    actor_ar = ActorAR(copy.deepcopy(CFG), jepa)
    actor_ar.train()

    def t_ar_step():
        loss = actor_ar.training_step(batch, 0)
        finite(loss, 'actor-ar loss')
        loss.backward()
    check('autoregressive training_step', t_ar_step)

    def t_ar_backtest():
        actor_ar.eval()
        with torch.no_grad():
            b = actor_ar.backtest(batch)
        # ActorAR inherits Actor.backtest, so it is scored over the same span
        # (eval ctx_len = 1), not over the shorter training rollout.
        eq(b['action'].shape, (B, S - 1, 1), 'AR actions')
        eq(b['equity'].shape, (B, S - 1), 'AR equity')
        eq(b['return_prob'].shape, (B, S - 1, RET_BINS), 'AR return probabilities')
        actor_ar.train()
    check('autoregressive backtest', t_ar_backtest)

    print('\nmetrics')
    import numpy as np
    import metrics as M

    K, N = 61, 4000
    rng = np.random.default_rng(0)
    idx = rng.integers(0, K - 31, N)
    onehot = np.eye(K)[idx]

    def t_scoring():
        assert abs(M.nll(onehot, idx)) < 1e-6, 'a perfect predictor has zero NLL'
        assert abs(M.bits_gained(onehot, idx, K) - np.log2(K)) < 0.2, 'perfect model should gain log2(K) bits'
        marg = np.tile(np.bincount(idx, minlength=K) / N, (N, 1))
        assert abs(M.bits_gained(marg, idx, K)) < 0.05, 'the marginal must gain ~0 bits over itself'
    check('NLL / bits_gained against the marginal baseline', t_scoring)

    def t_rps():
        near, far = np.eye(K)[idx + 1], np.eye(K)[idx + 30]
        assert M.rps(near, idx) < M.rps(far, idx), 'RPS must reward being close'
        assert M.top_k_accuracy(near, idx, 1) == M.top_k_accuracy(far, idx, 1) == 0.0
    check('RPS distinguishes a near miss from a far one', t_rps)

    def t_erank():
        assert abs(M.effective_rank(rng.normal(size=(2000, 32))) - 32) < 3
        assert M.effective_rank(np.ones((100, 32))) == 0.0
    check('effective_rank spans isotropic to collapsed', t_erank)

    def t_trading():
        r = np.full((20, 32), 0.01)
        e = np.cumprod(1 + r, axis=1)
        assert np.allclose(M.step_returns(e), r), 'step_returns must invert equity'
        assert M.max_drawdown(e).max() < 1e-12, 'a monotone-up curve has no drawdown'
        down = np.cumprod(1 + np.full((1, 10), -0.05), axis=1)
        assert abs(M.max_drawdown(down)[0] - (1 - 0.95 ** 10)) < 1e-9
        assert M.turnover(np.zeros((5, 10))) == 0.0
        assert abs(M.turnover(np.ones((5, 10))) - 0.1) < 1e-12, 'open-and-hold turns over once'
    check('equity, drawdown and turnover', t_trading)

    def t_summarize():
        e = np.cumprod(1 + np.full((20, 32), 0.01), axis=1)
        s = M.summarize(e, np.ones((20, 32)), 0.0005, 'x')
        assert abs(s['roi_mean'] - (e[0, -1] - 1)) < 1e-9
        assert s['win_rate'] == 1.0
        lo, hi = M.bootstrap_ci(np.array([1.0] * 50))
        assert abs(hi - lo) < 1e-9, 'a constant sample has a degenerate CI'
    check('summarize + bootstrap CI', t_summarize)

    print(f'\n{len(PASS)} passed, {len(FAIL)} failed')
    if FAIL:
        for name, exc in FAIL:
            print(f'  FAILED {name}: {type(exc).__name__}: {exc}')
        raise SystemExit(1)
    print('All smoke checks passed.')


if __name__ == '__main__':
    main()
