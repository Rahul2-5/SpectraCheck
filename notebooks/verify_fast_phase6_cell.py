# Complete Phase 6 verification after the existing worker exits; no training restart.
from pathlib import Path
from datetime import datetime, timezone
import time, json, sys, os, hashlib
PROJECT_DIR=Path('/content/ai-image-fft-detector')
root=Path('/content/drive/MyDrive/aidetect/r2')
checkpoint=root/'r2_fast_rf_t4_phase6_checkpoints'
previous=None
while fast_process.poll() is None:
    progress=json.loads((checkpoint/'status.json').read_text())
    cm=json.loads((checkpoint/'manifest.json').read_text())
    milestone=(progress['stage'],progress['current_model'],len(progress['completed_models']),len(cm.get('search_entries',{})))
    if milestone != previous:
        print(datetime.now(timezone.utc).isoformat(), 'Stage/model/completed/candidates:',milestone,flush=True)
        previous=milestone
    time.sleep(10)
assert fast_process.returncode == 0, FAST_LAUNCH_LOG.read_text()[-8000:]
print('Training worker completed. Verifying outputs and validation metrics...',flush=True)
os.chdir(PROJECT_DIR)
sys.path.insert(0,str(PROJECT_DIR/'src'))
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
from aidetect.io_utils import load_config,config_hash,sha256_of_file
from aidetect.train import load_training_features,make_frequency_matrix,make_pixel_matrix,make_stratified_folds,_folds_digest,_metrics,_checkpoint_contract
config=PROJECT_DIR/'config_r2_fast_rf_t4.yaml'
cfg=load_config(config)
artifacts=PROJECT_DIR/cfg['paths']['artifacts_root']
reports=PROJECT_DIR/cfg['paths']['reports_root']
review=sorted((root/'review_backup').glob('r2_fast_rf_t4_training_*'))[-1]
start=json.loads((review/'start.json').read_text())
assert start['status']=='completed'
assert start['config_sha256']==config_hash(cfg)=='46dbd2c248195aae6eef02a2219a56747090b4bb5f88e3cfb372ced68515766f'
manifest=json.loads((checkpoint/'manifest.json').read_text())
assert manifest['complete'] and len(manifest['entries'])==7 and len(manifest['search_entries'])==30
assert not (checkpoint/'running.lock').exists() and not (root/'fast_training_launch.lock').exists()
for folder in (artifacts,PROJECT_DIR/'artifacts/r2'):
    assert not (folder/'FREEZE.lock').exists() and not (folder/'EVALUATION_STARTED.json').exists()
for name,digest in manifest['entries'].items():
    receipt=checkpoint/(name+'.json')
    assert sha256_of_file(receipt)==digest,name
    entry=json.loads(receipt.read_text())
    assert entry['contract_sha256']==manifest['contract_sha256']
    if name!='feature_choice': assert sha256_of_file(checkpoint/(name+'.joblib'))==entry['model_sha256']
for name,digest in manifest['search_entries'].items():
    receipt=checkpoint/'search'/(name+'.json')
    assert sha256_of_file(receipt)==digest
    item=json.loads(receipt.read_text())
    assert item['contract_sha256']==manifest['contract_sha256'] and len(item['payload']['fold_scores'])==3
assert sha256_of_file(checkpoint/'migration.json')==manifest['migration_sha256']
lineage=json.loads((checkpoint/'migration.json').read_text())
assert set(lineage['reused_models'])=={'frequency_svm','frequency_xgb','pixel_svm'}
assert lineage['rf_models_and_candidates_reused'] is False
parent_checkpoint=root/'r2_fast_t4_phase6_checkpoints'
assert sha256_of_file(parent_checkpoint/'manifest.json')==lineage['source_manifest_sha256']
parent_manifest=json.loads((parent_checkpoint/'manifest.json').read_text())
assert parent_manifest['contract_sha256']==lineage['source_contract_sha256']==cfg['training']['reuse_source_contract_sha256']
reused_candidates_verified=0
for key in lineage['reused_models']:
    receipt=parent_checkpoint/(key+'.json')
    assert sha256_of_file(receipt)==parent_manifest['entries'][key]
    item=json.loads(receipt.read_text())
    assert item['contract_sha256']==lineage['source_contract_sha256']
    assert sha256_of_file(parent_checkpoint/(key+'.joblib'))==item['model_sha256']
    for index in range(10):
        name=f'{key}_{index:04d}'
        path=parent_checkpoint/'search'/(name+'.json')
        assert sha256_of_file(path)==parent_manifest['search_entries'][name]
        candidate=json.loads(path.read_text())
        assert candidate['contract_sha256']==lineage['source_contract_sha256'] and len(candidate['payload']['fold_scores'])==3
        reused_candidates_verified+=1
assert reused_candidates_verified==30
backup_hashes=json.loads((review/'sha256.json').read_text())
for name,digest in backup_hashes.items():
    assert sha256_of_file(review/name)==digest,name
for name,digest in manifest['published'].items(): assert sha256_of_file(Path(name))==digest,name
result=json.loads((artifacts/'training_results.json').read_text())
assert result['train_rows']==3500 and result['validation_rows']==750
assert result['cv_folds']==3 and result['search_iterations_per_model']==10 and result['seed']==42
assert result['test_features_read'] is False and result['execution']['xgb_device']=='cuda'
assert result['execution']['xgb_search_workers']==1 and result['execution']['search_workers']==2
train,val,provenance=load_training_features(cfg,config_file=config)
for key in provenance: assert result[key]==provenance[key],key
folds=make_stratified_folds(train['label'],folds=3,seed=42)
assert result['cv_splits_sha256']==_folds_digest(folds)
assert manifest['contract']==_checkpoint_contract(cfg,provenance,folds)
parent_config=PROJECT_DIR/'config_r2_fast_t4.yaml'
parent_cfg=load_config(parent_config)
parent_provenance={**provenance,'config_sha256':config_hash(parent_cfg),'config_file':str(parent_config)}
assert parent_manifest['contract']==_checkpoint_contract(parent_cfg,parent_provenance,folds)
sys.path.insert(0,str(PROJECT_DIR))
from scripts.migrate_fast_rf_checkpoints import validate_amendment
validate_amendment(parent_cfg,cfg)
assert manifest['contract_sha256']==config_hash(manifest['contract'])
choice=result['feature_choice']
assert len(choice['candidates'])==75
frequency,names=make_frequency_matrix(val,cfg,r0=choice['r0'],k_mad=choice['k_mad'])
pixels,pixel_names=make_pixel_matrix(val,cfg)
expected_keys={f'{feature}_{model}' for feature in ('frequency','pixel') for model in ('svm','rf','xgb')}
assert {m['key'] for m in result['models']}==expected_keys
metrics_verified=[]
with threadpool_limits(limits=1):
    for item in result['models']:
        path=PROJECT_DIR/item['model_path']
        assert sha256_of_file(path)==backup_hashes['artifacts/models/'+path.name]
        fitted=joblib.load(path)
        if item['model']=='rf':
            forest=fitted.named_steps['classifier']
            assert 100<=forest.n_estimators<=300 and forest.max_features=='sqrt'
        if item['model']=='xgb': assert fitted.named_steps['classifier'].get_params()['device']=='cuda'
        matrix=frequency if item['feature_set']=='frequency' else pixels
        fresh=_metrics(fitted,matrix,val['label'].to_numpy())
        for metric,value in fresh.items(): assert np.isclose(value,item[metric],rtol=0,atol=1e-12),(item['key'],metric)
        metrics_verified.append({'key':item['key'],**fresh})
        print('Validation metrics reproduced:',item['key'],fresh,flush=True)
frequency_results=[m for m in result['models'] if m['feature_set']=='frequency']
tie_order=cfg['training']['primary_model_tie_order']
primary=min(frequency_results,key=lambda m:(-m['validation_accuracy'],tie_order.index(m['model'])))
assert result['primary_model']['artifact_key']==primary['key']
assert sha256_of_file(Path(cfg['paths']['data_root'])/'splits.csv')==result['splits_sha256']
checks={'status':'completed_and_independently_verified','phase':6,'verified_utc':datetime.now(timezone.utc).isoformat(),
    'profile':result['training_profile'],'config_sha256':result['config_sha256'],'splits_sha256':result['splits_sha256'],
    'checkpoint_contract_sha256':manifest['contract_sha256'],'training_started_utc':start['started_utc'],'training_completed_utc':start['completed_utc'],
    'training_elapsed_seconds':(datetime.fromisoformat(start['completed_utc'])-datetime.fromisoformat(start['started_utc'])).total_seconds(),
    'train_rows':3500,'validation_rows':750,'cv_folds':3,'search_candidates_per_model':10,'candidate_receipts_verified':60,'new_candidate_receipts_verified':30,'reused_candidate_receipts_verified':reused_candidates_verified,'model_reuse':lineage,
    'model_receipts_verified':6,'backup_files_verified':len(backup_hashes),'validation_metrics_reproduced':metrics_verified,
    'feature_choice':{'r0':choice['r0'],'k_mad':choice['k_mad']},'primary_model':primary['key'],
    'primary_validation_accuracy':primary['validation_accuracy'],'execution':result['execution'],
    'test_features_read':False,'freeze_or_evaluation_started':False,'supervisor_signoff':'unverified','drive_backup':str(review)}
original_start=json.loads((root/'review_backup/r2_fast_t4_training_20261031T085807Z/start.json').read_text())
checks['workflow_started_utc']=original_start['started_utc']
checks['overall_elapsed_seconds']=(datetime.fromisoformat(start['completed_utc'])-datetime.fromisoformat(original_start['started_utc'])).total_seconds()
check_path=review/'phase_6_completion_checks.json'
with check_path.open('x') as stream: json.dump(checks,stream,indent=2,sort_keys=True)
print('PHASE 6 COMPLETED AND VERIFIED',json.dumps(checks,indent=2),flush=True)
print('Completion record SHA-256:',sha256_of_file(check_path),flush=True)
print('STOP: Phase 7 is not authorized. Test values remain sealed.',flush=True)
