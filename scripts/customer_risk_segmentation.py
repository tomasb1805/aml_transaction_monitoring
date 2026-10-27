"""
Customer risk segmentation — the AML counterpart of the credit-risk personas.

In AML, a customer's risk rating is not static: it is refreshed from KYC data and
observed behaviour. Here the originator of every transaction is assigned a segment
from its behavioural profile as known on the transaction date (prior days only),
so the segment is point-in-time and leak-free.
"""
import numpy as np
import pandas as pd


def assign_segment(df: pd.DataFrame) -> pd.Series:
    """
    Assigns a risk segment to the originator of each transaction (vectorised rule hierarchy).

    Segment hierarchy:
        new_relationship   -  no prior activity on file: behaviour cannot be assessed yet.
        crypto_user        -  over 25% of prior outgoing value paid in Bitcoin.
        cash_intensive     -  over 50% of prior outgoing value paid in cash.
        pass_through       -  money out ≈ money in (spend-to-income 0.9-1.1): funnel-account profile.
        high_degree_hub    -  100+ distinct counterparties paid since first seen.
        net_spender        -  outflows exceeded inflows on more than half of the observed days.
        mainstream         -  all other customers.
    """
    no_history = df['orig_prior_days_observed'].fillna(0) == 0

    conditions = [
        no_history,
        df['orig_crypto_outflow_ratio'] > 0.25,
        df['orig_cash_withdrawal_ratio'] > 0.50,
        df['orig_spend_to_income_ratio'].between(0.9, 1.1),
        df['orig_out_distinct_cp_cum'] >= 100,
        df['orig_days_net_negative'] > 0.5 * df['orig_prior_days_observed'],
    ]
    choices = ['new_relationship', 'crypto_user', 'cash_intensive',
               'pass_through', 'high_degree_hub', 'net_spender']

    return pd.Series(np.select(conditions, choices, default='mainstream'), index=df.index)


def refine_mainstream(df: pd.DataFrame) -> pd.Series:
    """
    Refinement of the 'mainstream' segment.
    Separates it into sub-groups using the KYC entity type of the originator.
    """
    entity = df['orig_entity_type'].where(
        df['orig_entity_type'].isin(['corporation', 'partnership', 'sole_proprietorship']),
        'other'
    )
    return 'mainstream_' + entity


def assign_customer_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the 'customer_segment' column, with the 'mainstream' segment refined.
    """
    df = df.copy()
    segment = assign_segment(df)
    mask = segment == 'mainstream'

    segment[mask] = refine_mainstream(df[mask])
    df['customer_segment'] = segment

    return df
