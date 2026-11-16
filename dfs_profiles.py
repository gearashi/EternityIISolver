"""Named pruning profiles shared by local DFS, the CLI and the dashboard."""

PRUNING_PROFILES = ('baseline', 'strong', 'full', 'full-pivot', 'full-dispatch', 'adaptive', 'components', 'joint', 'relational')


def dfs_profile_options(profile):
    """Return a fresh option set shared by packaged DFS and offline replays."""
    if profile not in PRUNING_PROFILES:
        raise ValueError('Unknown DFS pruning profile')
    if profile == 'baseline':
        return {}
    if profile == 'components':
        return {'component_completion': True}
    if profile == 'joint':
        return {'extra_rules': {'joint_colors': True}}
    if profile == 'relational':
        return {'extra_rules': {'relational': True}}
    options = dict(gac=True, independent_completion=True, dual_branch=True,
                   edge_branch=True, extra_rules=None)
    if profile in ('full', 'full-pivot', 'full-dispatch', 'adaptive'):
        options.update(probe=True, extra_rules={})
    if profile in ('full-pivot', 'full-dispatch'):
        options['pivot_cut_v1'] = True
    if profile == 'full-dispatch':
        options['pivot_dispatch_v1'] = True
    if profile == 'adaptive':
        options['adaptive'] = True
    return options


def validate_pivot_engine(profile, engine, workers):
    if profile in ('full-pivot', 'full-dispatch') and (engine != 'dfs' or workers != 1):
        raise ValueError('Full filtering with a pivot proof requires standard DFS with one CPU worker')
