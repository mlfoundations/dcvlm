import json
import logging
from internvl.train.constants import NUMERICAL_COMPARATORS, MEMBERSHIP_COMPARATORS

def decode_annotations(sample: dict, key: str, logger: logging.Logger = None) -> dict:
    # empty check cuz some samples might fail during annot
    annots = sample.get(key, {}) 
    if not isinstance(annots, dict):
        try:
            annots = json.loads(annots)
        except Exception as e:
            if logger is not None:
                logger.error(
                    f"Could not decode '{key}' for sample key={sample.get('__key__')} from {sample.get('__ds_name__')}. "
                    f"Error: {repr(e)}"
                )
            return {}
    return annots


def _is_valid_numerical(filter_spec: dict) -> bool:
    """Check if the filter requires numerical comparison. Such a filter must have:
    - a 'threshold' key with a numerical value;
    - a 'sign' key with one of the following values: ">", ">=", "<", "<=", "=="
    """
    return "threshold" in filter_spec and filter_spec["sign"] in NUMERICAL_COMPARATORS


def _run_numerical(
    filter_spec: dict, 
    filter_name: str, 
    sample_annotations: dict, 
    sample: dict, 
    logger: logging.Logger = None
) -> bool:
    # base case: we have a None value for the annotation
    # this means the filter is not applicable, so we keep the sample
    # (e.g., IQA is None for text-only datasets)
    if sample_annotations[filter_name] is None:
        return True
    # grab the exact threshold and sign for the current online filter
    # NOTE: all thresholds have to be predetermined and computed offline.
    # These are *NOT* percentiles but rather absolute values.
    threshold, sign = filter_spec["threshold"], filter_spec["sign"]

    # if sample doesn't satisfy condition, we filter it out!
    comparator = NUMERICAL_COMPARATORS[sign]
    if not comparator(sample_annotations[filter_name], threshold):
        if logger is not None:
            logger.error(
                f"Sample {sample.get('__key__')} from {sample.get('__ds_name__')} filtered out by '{filter_name}'."
                f" {sample_annotations[filter_name]} {sign} {threshold} is False."
            )
        return False
    return True


def _is_valid_membership(filter_spec: dict) -> bool:
    """Check if the filter requires membership comparison. Such a filter must have:
    - a 'values' key with a list of values (the set to check against);
    - a 'sign' key with one of the following values: "in", "not in"
    """
    return (
        "values" in filter_spec 
        and filter_spec["sign"] in MEMBERSHIP_COMPARATORS
        and isinstance(filter_spec["values"], list)
    )


def _run_membership(
    filter_spec: dict, 
    filter_name: str, 
    sample_annotations: dict, 
    sample: dict, 
    logger: logging.Logger = None
) -> bool:
    # grab values and sign for the membership comparison
    values, sign = filter_spec["values"], filter_spec["sign"]

    # if sample doesn't satisfy condition, we filter it out!
    comparator = MEMBERSHIP_COMPARATORS[sign]
    if not comparator(sample_annotations[filter_name], values):
        if logger is not None:
            logger.error(
                f"Sample {sample.get('__key__')} from {sample.get('__ds_name__')} filtered out by '{filter_name}'."
                f" {sample_annotations[filter_name]} {sign} {values} is False."
            )
        return False
    return True


def evaluate_conditions(
    sample: dict, 
    sample_annotations: dict, 
    conditions: dict,
    logger,
    filter_name: str,
    verbose: bool = False,
) -> bool:
    conditions_met = True
    for cond_key, spec_list in conditions.items():
        for spec in spec_list:
            if _is_valid_numerical(spec):
                if not _run_numerical(spec, cond_key, sample_annotations, sample, logger if verbose else None):
                    conditions_met = False
                    if verbose:
                        logger.error(
                            f"Not applying filter {filter_name} to {sample['__key__']} x {sample['__url__']} "
                            f"due to missing conditions on {cond_key}"
                        )
                    break
            elif _is_valid_membership(spec):
                if not _run_membership(spec, cond_key, sample_annotations, sample, logger if verbose else None):
                    conditions_met = False
                    if verbose:
                        logger.error(
                            f"Not applying filter {filter_name} to {sample['__key__']} x {sample['__url__']} "
                            f"due to missing conditions on {cond_key}"
                        )
                    break
        if not conditions_met:
            break
    return conditions_met


def online_filter(
    sample: dict, 
    online_filters: dict[str, dict[str, float]],
    filter_conditions: dict[str, dict[str, dict[str, float]]],
    verbose: bool = False
) -> bool:
    """Return True if the sample passes ALL online filters; otherwise False."""

    logger = logging.getLogger()

    # if no filter, keep the sample
    if len(online_filters) == 0:
        return True

    # if filters doesnt exist in sample, we raise
    if ("filters.json" not in sample) and ("metadata.json" not in sample):
        if verbose:
            logger.error(
                f"Neither 'filters.json' nor 'metadata.json' exist. Cannot run any filtering. "
                f"Sample key={sample.get('__key__')} from {sample.get('__ds_name__')}"
            )
        return False
    
    # grab the annotations for current sample
    # NOTE: designed s.t. "metadata" rules over "filters"
    sample_annotations = {}
    for annot_key in ('filters.json', 'metadata.json'):
        annots = decode_annotations(sample, annot_key, logger if verbose else None)
        sample_annotations.update(annots)
    if len(sample_annotations) == 0:
        if verbose:
            logger.error(
                f"Sample key={sample.get('__key__')} from {sample.get('__ds_name__')} has no annotations. "
                "Skipping it."
            )
        return False
    
    # NOTE: added to support dataset-specific filtering
    # when collating many datasets into a single entry
    nested = online_filters.get("nested", False)
    if nested:
        ds_name = sample.get('__ds_name__')
        specs_to_check = online_filters.get(ds_name, {})
    else:
        specs_to_check = online_filters

    # This filtering does AND over all conditions in online_filters,
    # we iterate over each one and return False the minute we encounter a failed condition
    for filter_name, filter_spec in specs_to_check.items():

        # if the filter name from online_filters doesn't exist in current sample, we raise
        # FIXME: Maybe we shouldn't raise here? e.g.:
        # - image datasets won't have 'textdecontamination'
        # - text datasets won't have 'clip_score', 'iqa', etc
        # should we just skip the filter instead?
        if filter_name not in sample_annotations:
            if verbose:
                logger.error(f"{filter_name} does not exist in the annotations of this dataset, {sample.get('__ds_name__')}")
            continue
            # return False

        # evaluate the conditions for this filter 
        # (e.g., maybe we wanna apply it to english-only data for instance)
        if nested:
            conditions = filter_conditions.get(ds_name, {}).get(filter_name, None)
        else:
            conditions = filter_conditions.get(filter_name, None) 
        if conditions is None:
            # no conditions, always apply the filter
            conditions_met = True 
        else:
            # if there are conditions, make sure they're all true or don't use this filter
            # since it's subject to such conditions
            conditions_met = evaluate_conditions(
                sample, sample_annotations, conditions, 
                logger, filter_name, verbose
            )
        # skip this filter if conditions aren't met
        if not conditions_met: 
            continue

        # support a list of specs for a given annotation
        # this is useful to keep samples within a range of values
        # (e.g. [{"threshold": 0.3, "sign": ">="}, {"threshold": 0.5, "sign": "<"}])
        if isinstance(filter_spec, dict):
            filter_spec = [filter_spec]
        for item in filter_spec:
            if _is_valid_numerical(item):
                if not _run_numerical(item, filter_name, sample_annotations, sample, logger if verbose else None):
                    return False
            elif _is_valid_membership(item):
                if not _run_membership(item, filter_name, sample_annotations, sample, logger if verbose else None):
                    return False
            else:
                # if we hit an invalid spec, we log and skip it
                logger.error(f"Invalid filter spec {item} for filter '{filter_name}'")
                continue

    # if all conditions are satisifed, we keep the sample
    return True