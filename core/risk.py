def risk_ok(daily_pnl: float, start_balance: float, max_daily_loss: float, consecutive_losses: int, max_consecutive_losses: int) -> bool:
    max_loss = -start_balance * max_daily_loss
    return daily_pnl > max_loss and consecutive_losses < max_consecutive_losses

def position_size(start_balance: float, risk_per_trade: float, sl_distance: float) -> float:
    risk_usd = start_balance * risk_per_trade
    if sl_distance <= 0:
        return 0.0
    return max(risk_usd / sl_distance, 0.001)