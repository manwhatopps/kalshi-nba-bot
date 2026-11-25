"""QuantSport: the investment layer on top of the Kalshi research bot in `bot/`.

    scan -> estimate -> logical chain -> risk audit -> four tiers -> paper book -> settlement audit -> dashboard

Nothing here places an order. `quantsport/execution.py` is the only module that could, and it refuses without
a one-use approval bound to the exact order, signed with a key that lives outside this repository.
"""
MODEL_VERSION = "qs-0.1"
