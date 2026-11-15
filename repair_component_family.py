"""Build a complete, scoped component family for the native repair-radius worker."""
from time import monotonic
from repair_components import screen


def prepare(problem, parent, prices, radius, remaining_work, deadline):
    # One counted visit per orientation/mask, shape, extension attempt or
    # recipient/origin lookup. A shape has at most k*k origin lookups and
    # 4*k possible extensions. This gives the screen a safe shape allocation
    # before performing any work against an explicit caller budget.
    setup_bound = 16 * sum(max(0,len(d)-1) for d in problem.domains)
    per_shape = 1 + radius * radius + 4 * radius
    shape_cap = min(400000, (remaining_work - setup_bound) // per_shape)
    if shape_cap < 1:
        return {'status':'work_limit', 'work':0, 'minimum_setup_budget':setup_bound+per_shape}, ''
    expired = lambda: monotonic() >= deadline
    if expired():
        return {'status':'timeout','work':0}, ''
    details = screen(problem,parent,prices,radius=radius,max_shapes=shape_cap,keep=1024,
                     should_stop=expired)
    shapes = details.pop('samples')
    work = sum(details[k] for k in ('shapes','extensions','recipient_origin_visits',
                                    'orientation_mask_scores'))
    if work > remaining_work:
        raise RuntimeError('component screen exceeded its counted allocation')
    count = sum(details['viable_by_size'].values())
    summary = dict(counts=details,work=work,positive_shapes=count)
    if expired():
        return dict(summary,status='timeout'), ''
    if details['status'] != 'complete':
        return dict(summary,status='work_limit' if shape_cap<400000 else 'component_cover_limit'), ''
    if len(shapes)!=count:
        return dict(summary,status='component_cover_limit'), ''
    if len({tuple(item['cells']) for item in shapes})!=count:
        raise RuntimeError('repeated connected component')
    rows = [str(count)]
    serial_work = 1 + sum(2+len(s['cells']) for s in shapes)
    if work+serial_work > remaining_work:
        return dict(summary,status='work_limit'), ''
    summary['work'] += serial_work
    summary['serialization_work'] = serial_work
    for item in shapes:
        cells = item['cells']
        rows.append(' '.join(map(str,(len(cells),item['bound'],*cells))))
    return dict(summary,status='complete'), '\n'.join(rows)+'\n'
