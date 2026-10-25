"""Exclusive single-shot lockbox. Never called automatically by the development runner."""
import json,os,datetime,operator,traceback
from pathlib import Path
from .config import ART,parser,dump,config,sha
from .data import load
from .folds import manifest
from .train_eval_walkforward import run_year,verify_freeze
from .evaluate import report
from .audit_modeling import audit
OPS={'<':operator.lt,'<=':operator.le,'>':operator.gt,'>=':operator.ge,'==':operator.eq}

def criteria_check(spec,results):
    out={}
    for name,ruleset in spec.items():
        rules=[]
        for rule in ruleset['rules']:
            value=results
            keys=rule['path'][1:].split('/') if rule['path'].startswith('/') else rule['path'].split('.')
            for key in keys:
                key=key.replace('~1','/').replace('~0','~')
                value=value[int(key)] if isinstance(value,list) else value[key]
            passed=value is not None and bool(OPS[rule['op']](value,rule['threshold']))
            rules.append(dict(**rule,observed=value,passed=passed))
        out[name]=dict(description=ruleset['description'],rules=rules,passed=all(r['passed'] for r in rules))
    return out

def main():
    p=parser();p.add_argument('--execute-once',action='store_true');p.add_argument('--criteria',type=Path,required=True);a=p.parse_args()
    if not a.execute_once:raise SystemExit('Requires explicit --execute-once authorization')
    if a.allow_sparse_research:raise SystemExit('Official lockbox is forbidden in sparse-history research mode')
    criteria=json.loads(a.criteria.read_text())
    if set(criteria)!={f'S{i}' for i in range(1,7)}:raise ValueError('Supply authoritative S1-S6 criteria, not guessed thresholds')
    for name,rs in criteria.items():
        if not rs.get('description') or not rs.get('rules'):raise ValueError(f'{name} has no approved rules')
        for rule in rs['rules']:
            if rule['op'] not in OPS or not isinstance(rule['threshold'],(int,float)):raise ValueError('Invalid acceptance rule')
    qa=json.loads((ART/'evaluation_qa_report.json').read_text())
    if not qa['strict_specification_QA_pass']:raise RuntimeError('Evaluation strict QA must pass before opening lockbox')
    frozen=config();verify_freeze(frozen,a.data)
    # Verify rule paths before consuming the lockbox, against the same report schema.
    criteria_check(criteria,json.loads((ART/'evaluation_metrics.json').read_text()))
    marker=ART/'LOCKBOX_CONSUMED.json'
    event=dict(status='STARTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               data_sha256=sha(a.data),frozen_config_sha256=sha(ART/'frozen_config_m3.json'),
               criteria_sha256=sha(a.criteria),code_sha256=frozen['code_sha256'])
    # x is atomic exclusive creation: failures remain consumed; there is NO --force flag.
    with marker.open('x') as f:json.dump(event,f,indent=2);f.flush();os.fsync(f.fileno())
    try:
        d=load(a.data,False,lockbox=True)
        dump(ART/'lockbox_fold_manifest.json',manifest(d,include_lockbox=True))
        event['lockbox_fold_manifest_sha256']=sha(ART/'lockbox_fold_manifest.json')
        run_year(d,2026,frozen,'lockbox');results=report('lockbox')
        checks=criteria_check(criteria,results);q=audit(a.data,False,'lockbox')
        event['status']='COMPLETED';event['acceptance_pass']=all(v['passed'] for v in checks.values()) and q['strict_specification_QA_pass']
        dump(ART/'lockbox_acceptance.json',checks)
        lines=['# NQ4H-Q Final Evaluation Report','',f"Acceptance: {'PASS' if event['acceptance_pass'] else 'FAIL'}",'',
               '## Frozen evaluation (2019–2025)','See EVALUATION_REPORT.md and evaluation_metrics.json.','',
               '## Single-shot lockbox (2026)','See LOCKBOX_REPORT.md and lockbox_metrics.json.','']
        for name,r in checks.items():lines.append(f"- {name}: {'PASS' if r['passed'] else 'FAIL'} — {r['description']}")
        lines+=['',f"Dataset SHA-256: `{event['data_sha256']}`",f"Frozen config SHA-256: `{event['frozen_config_sha256']}`",
                'Metrics are computed in raw log-return units. Inactive confidence is NaN by design.']
        (ART/'FINAL_EVALUATION_REPORT.md').write_text('\n'.join(lines)+'\n')
        card=['# NQ4H-Q Model Card','', '## Intended use','Research-only CME NQ 4-hour return distribution forecasting; not investment advice.',
              '', '## Architecture','Independent vectorized PatchTST-Q and two-layer LSTM deep ensembles. Seven noncrossing quantiles.',
              '', '## Data and evaluation',f"Dataset SHA-256: `{event['data_sha256']}`. Expanding folds, 2019–2025 evaluation; single-shot 2026 lockbox.",
              '', '## Limitations','Midquote return backtests are not fill simulations. Event gaps, structural breaks, model selection and serial dependence affect inference. Conformal exchangeability is not guaranteed in financial time series. CQR rearrangement and ACI do not provide unconditional finite-sample coverage guarantees after adaptation.',
              '',f"Acceptance passed: {event['acceptance_pass']}. See ARTIFACTS/FINAL_EVALUATION_REPORT.md and lockbox_acceptance.json."]
        (ART.parent/'MODEL_CARD.md').write_text('\n'.join(card)+'\n')
        if not event['acceptance_pass']:raise RuntimeError('Lockbox completed but acceptance failed; never rerun/tune against it')
    except BaseException as e:
        event['status']='FAILED_OR_ACCEPTANCE_REJECTED';event['error']=str(e)
        dump(ART/'lockbox_error.json',dict(traceback=traceback.format_exc()))
        raise
    finally:
        event['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();dump(marker,event)
if __name__=='__main__':main()
