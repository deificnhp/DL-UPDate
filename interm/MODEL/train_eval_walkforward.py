import json
from pathlib import Path

from .config import parser,ART,ROOT,dump,config,sha,tracked_sources
from .data import load
from .folds import make_fold
from .gpu import fit
from .predictions import calibrated
from .baselines import run_fold
from .evaluate import report

def _resolve_code_path(rel):
    """Resolve a frozen code path to a file on disk.
    New freezes use repo-root-relative POSIX paths ('MODEL/data.py', 'FE/nq4hq_features/pipeline.py').
    Legacy freezes used bare file names relative to the MODEL package ('data.py')."""
    p=Path(rel)
    if p.is_absolute(): return p
    for cand in (ROOT/rel,ART.parent/rel):
        if cand.is_file(): return cand
    return None

def verify_freeze(frozen,data):
    """Fail-closed verification of the M3 freeze (F-2).
    Every check MUST pass; the function raises RuntimeError on the first violation.
    There is no bypass, no override flag and no 'return' short-circuit: if the dataset,
    the configs, the fold ledger or ANY of the frozen source files differ from what was
    frozen, evaluation is not allowed to run."""
    if not isinstance(frozen,dict): raise RuntimeError('M3 freeze is not a mapping')
    code=frozen.get('code_sha256')
    if not isinstance(code,dict) or not code: raise RuntimeError('M3 freeze has no code_sha256 manifest')
    # 1. dataset bytes
    got=sha(data)
    if got!=frozen.get('dataset_sha256'):
        raise RuntimeError(f"Dataset hash differs from M3: frozen {frozen.get('dataset_sha256')}, actual {got}")
    # 2. baseline config
    bp=ART/'baseline_config.json'
    if not bp.is_file(): raise RuntimeError('Baseline config missing; cannot verify M3 freeze')
    got=sha(bp)
    if got!=frozen.get('baseline_config_sha256'):
        raise RuntimeError(f"Baseline config differs from M3: frozen {frozen.get('baseline_config_sha256')}, actual {got}")
    # 3. fold ledger
    fp=ART/'fold_manifest.json'
    if not fp.is_file(): raise RuntimeError('Fold manifest missing; cannot verify M3 freeze')
    got=sha(fp)
    if got!=frozen.get('fold_manifest_sha256'):
        raise RuntimeError(f"Fold ledger differs from M3: frozen {frozen.get('fold_manifest_sha256')}, actual {got}")
    # 4. every frozen source file
    for rel,digest in sorted(code.items()):
        p=_resolve_code_path(rel)
        if p is None: raise RuntimeError(f'Code changed after freeze: {rel} (file missing from the repository)')
        if sha(p)!=digest: raise RuntimeError(f'Code changed after freeze: {rel}')
    # 5. no untracked source file: every source file that exists now must be in the manifest,
    #    otherwise a new (or previously uncovered) file could change the results silently.
    covered={rel for rel in code if '/' in rel}
    uncovered=[r for r in (x.relative_to(ROOT).as_posix() for x in tracked_sources())
               if r not in covered and Path(r).name not in code]
    if uncovered:
        raise RuntimeError('M3 freeze does not cover current source files: '+', '.join(sorted(uncovered)))
    return True


def run_year(d,year,frozen,scope):
    parts=make_fold(d,year)
    bcfg=json.loads((ART/'baseline_config.json').read_text());run_fold(d,year,bcfg,scope)
    for arch,name in [('patchtst','PatchTST-Q'),('lstm','C1-LSTM')]:
        cfg=frozen['config'] if arch=='patchtst' else frozen['lstm_config']
        checkpoint=ART/'checkpoints'/f'{year}_{arch}.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
        outputs, meta = fit(d, parts, cfg, arch, checkpoint=checkpoint, enforce_memory_band=False)
        cmeta,trace=calibrated(d,parts,outputs,name,year,scope)
        dump(ART/'provenance'/f'{year}_{arch}.json',dict(fold=year,fit=meta,calibration=cmeta,aci_alpha_after_observation=trace))
        print('EVALUATED',year,name,flush=True)

def main():
    a=parser().parse_args();frozen=config();verify_freeze(frozen,a.data)
    d=load(a.data,a.allow_sparse_research)
    for year in range(2019,2026):run_year(d,year,frozen,'evaluation')
    report('evaluation')
if __name__=='__main__':main()
