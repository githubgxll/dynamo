# SPDX-License-Identifier: Apache-2.0
"""Verify versioned intent decisions before applying the existing IR contract."""
from copy import deepcopy
from .client import UpstreamRejected


def resolved_contract(normalized, expected, ir):
    try:
        return _resolve(normalized, expected, ir)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise UpstreamRejected('context_ir_intent_contract_mismatch') from exc


def _resolve(normalized, expected, ir):
    normalized, expected = deepcopy(normalized), deepcopy(expected)
    cfg = normalized['input_resolution']
    record = ir.get('provenance', {}).get('intent_resolution')
    if (not isinstance(record, dict) or record.get('version') != 1
            or record.get('status') not in ('resolved','defaulted','skipped')
            or not isinstance(record.get('decisions'),list)):
        raise ValueError('missing resolution evidence')
    refs = {e['ordinal']:e for e in normalized['references']}
    decisions = {}
    source_text=normalized['intent']+'\n'+'\n'.join(d['text'] for d in normalized.get('dialogue',[]))
    required=set()
    for idx,entry in refs.items():
        if cfg['workflow']=='fl2va' or entry['kind']=='audio':required.add((idx,'role'))
        if entry.get('has_audio'):required.add((idx,'soundtrack_policy'))
    for row in record['decisions']:
        if not isinstance(row,dict) or type(row.get('index')) is not int:
            raise ValueError('invalid decision')
        key=(row['index'],row.get('field'))
        if key not in required or key in decisions:raise ValueError('unexpected decision')
        entry=refs[row['index']];value=row.get('value');field=row['field']
        allowed=({'preserve','ignore'} if field=='soundtrack_policy' else
                 {'frame_anchor_first','frame_anchor_last'} if cfg['workflow']=='fl2va' else {'voice_timbre','bgm'})
        if value not in allowed:raise ValueError('invalid resolved value')
        explicit=entry.get('role_stated',True) if field=='role' else entry.get('soundtrack_policy') is not None
        source=row.get('source')
        if explicit:
            if value!=entry[field] or source!='parameter':raise ValueError('explicit option changed')
        elif source=='parameter':raise ValueError('default presented as parameter')
        elif source=='prompt':
            evidence=row.get('evidence')
            if not isinstance(evidence,str) or not evidence.strip() or evidence not in source_text:
                raise ValueError('missing source evidence')
        elif source=='default':
            fallback='preserve' if field=='soundtrack_policy' else 'voice_timbre' if entry['kind']=='audio' else 'frame_anchor_first'
            if value!=fallback:raise ValueError('incorrect fallback')
        elif source=='constraint':
            if cfg['workflow']!='fl2va' or len(refs)!=2:raise ValueError('invalid constraint')
        else:raise ValueError('invalid decision source')
        decisions[key]=row
    if set(decisions)!=required:raise ValueError('missing material decisions')
    for entry in expected:
        idx=entry['index']
        if (idx,'role') in decisions:
            entry['role']=decisions[(idx,'role')]['value'];entry['role_stated']=True
        if (idx,'soundtrack_policy') in decisions:
            entry['soundtrack_policy']=decisions[(idx,'soundtrack_policy')]['value']
    if cfg['workflow']=='fl2va':
        anchors=sorted(expected,key=lambda e:e['role']=='frame_anchor_last')
        if len(anchors)==2 and {e['role'] for e in anchors}!={'frame_anchor_first','frame_anchor_last'}:
            raise ValueError('invalid keyframe binding')
        for number,entry in enumerate(anchors,1):entry['labels']=[f'<Picture {number}>']
        normalized['mode']='fl2va' if len(anchors)==2 else 'l2va' if anchors[0]['role']=='frame_anchor_last' else 'i2va'
        indices=[0 if entry['role']=='frame_anchor_first' else -1 for entry in sorted(expected,key=lambda e:e['index'])]
        params=dict(normalized['generation_parameters']);params['frame_indices']=indices
        if cfg.get('adaptive_canvas'):
            canvas=cfg['anchor_canvases'][str(anchors[0]['index'])]
            normalized['canvas']=canvas
            params.pop('size',None);params.update(resolution='768P',ratio='adaptive')
        normalized['generation_parameters']=params
    warnings=[]
    for warning in record.get('warnings',[]):
        if not isinstance(warning,dict) or warning.get('code') not in {'intent_resolution_defaulted','intent_resolution_conflict'}:
            raise ValueError('invalid resolution warning')
        value={'code':warning['code']}
        if 'index' in warning:
            if type(warning['index']) is not int or warning['index'] not in refs:raise ValueError('invalid warning index')
            value['index']=warning['index']
        warnings.append(value)
    return normalized,expected,warnings
