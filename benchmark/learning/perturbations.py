"""Schedule fixed evaluation conditions and summarize their published results."""


def run_evaluation_looks(app, args, looks, run):
    """Run the same episode seeds per look; keep robustness and nominal scores separate."""
    from copy import copy
    from benchmark.io import write_json
    from benchmark.learning.evaluation import evaluate

    sweep = len(looks) > 1
    summary = {'status': 'running', 'sweep': True, 'looks': [],
               'episodes_per_look': args.episodes}
    if sweep:
        write_json(args.output_dir / 'results.json', summary)
    try:
        for name, look in looks:
            child = copy(args)
            child.perturbation = look
            child.output_dir = args.output_dir / name if sweep else args.output_dir
            if sweep:
                child.output_dir.mkdir()
            report = {'perturbation': look.to_dict(), 'look': name}
            if sweep:
                summary['looks'].append({'look': name, 'results': str(child.output_dir / 'results.json'),
                                         'perturbation': look.to_dict(), 'status': 'running'})
                write_json(args.output_dir / 'results.json', summary)
            try:
                evaluate(app, child, report, run=run)
            finally:
                if sweep:
                    summary['looks'][-1].update({key: report[key] for key in
                        ('status', 'score_complete', 'episodes_completed', 'successes',
                         'success_rate', 'completed_episode_success_rate', 'error') if key in report})
        summary['status'] = 'completed'
    except BaseException as error:
        summary.update(status='error', error=f'{type(error).__name__}: {error}')
        if sweep and summary['looks'] and summary['looks'][-1]['status'] == 'running':
            summary['looks'][-1].update(status='error', score_complete=False)
        raise
    finally:
        if sweep:
            write_json(args.output_dir / 'results.json', summary)
