from dataclasses import dataclass

@dataclass
class Config:
    symbol: str = "BTCUSDT"
    interval: str = "15m"
    limit: int = 800

    start_balance: float = 500.0
    leverage: int = 3

    risk_per_trade: float = 0.005
    max_daily_loss: float = 0.02
    max_consecutive_losses: int = 3
    cooldown_seconds: int = 60
    
    arm_status_path: str = "state/arm_status.json"
    atr_period: int = 14
    adx_period: int = 14
    atr_ma_period: int = 50

    # Regime thresholds
    adx_trend: float = 22.0
    atr_vol_mult: float = 1.8
    
    # Equity / risk intelligence
    max_drawdown: float = 0.05          # 5% DD bo'lsa defensive mode
    dd_risk_mult: float = 0.5           # DD bo'lsa risk 50% ga tushadi

    # Adaptive risk
    risk_mult_min: float = 0.25
    risk_mult_max: float = 1.25

    fee_rate: float = 0.0004
    slippage: float = 0.0002

    # Discipline / Fund params
    max_trades_per_day = 8
    max_stopouts_per_day = 3
    max_daily_loss_usd = 15.0
    max_daily_drawdown = 0.05
    vol_kill_atr_mult = 2.8

    # Journal
    journal_path = "state/journal.jsonl"

    # (agar yo'q bo'lsa)
    ht_ema_fast = 50
    ht_ema_slow = 200
    ht_adx_min = 18.0
    ms_lookback = 20
    ms_break_pct = 0.001

    ht_interval: str = "1h"
    ht_limit: int = 500
    
    # realism
    fee_rate = 0.0004
    slippage_bps = 2.0
    
    maker_fee = 0.0002
    taker_fee = 0.0004


    # confidence risk shaping
    conf_risk_min_mult = 0.60
    conf_risk_max_mult = 1.35

    # Arm Guard (auto-disable)
    arm_guard_path = "state/arm_guard.json"
    arm_disable_loss_streak = 4
    arm_disable_hours = 24.0
    
    # Global trend (1H)
    ht_ema_fast: int = 50
    ht_ema_slow: int = 200
    ht_adx_period: int = 14
    ht_adx_min: float = 18.0

    # Market structure (1H)
    ms_lookback: int = 20   # HH/HL yoki LL/LH aniqlash uchun
    ms_break_pct: float = 0.001  # 0.1% breakout/filter

    # Global bias kuchi (scoring)
    bias_strong_score: float = 2.5   # kuchli trend
    bias_weak_score: float = 1.5     # zaif trend

    loop_seconds: int = 30

    learner_path: str = "state/learner_state.json"
    trades_path: str = "state/trades.csv"
    