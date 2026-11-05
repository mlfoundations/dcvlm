import os
import json
import os.path as osp
import logging
import datetime
from shutil import copy, rmtree
from typing import Union
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

logger = logging.getLogger()

def is_transformers_compatbile(path: str, return_dict=False) -> Union[bool, dict]:
    """
    Check if the checkpoint at the given path is compatible with Hugging Face Transformers.
    This means:
    - There is a config.json file
    - The config.json file has the correct AutoMapping for InternVLChat
    - The configuration_internvl_chat.py and modeling_internvl_chat.py files are present
    - There is at least one .safetensors file

    If return_dict is True, return a dictionary with the results of each check.
    If return_dict is False, return True if all checks pass, False otherwise.
    """
    ret = {
        "config_ok": False,
        "safetensors_ok": False,
        "auto_map_ok": False,
        "pyfiles_ok": False
    }
    
    # assert there is a config.json
    config_path = osp.join(path, 'config.json')
    if not osp.isfile(config_path):
        logger.warning(
            f"No config.json found in {path}. "
            "Not a transformers compatible checkpoint."
        )
    else:
        ret["config_ok"] = True
    
    # load the config.json and check that it has AutoMapping
    if ret["config_ok"]:
        with open(config_path, 'r') as f:
            config = json.load(f)

        if 'auto_map' not in config:
            logger.warning(
                f"No 'auto_map' found in config.json in {path}. "
                "Not a transformers compatible checkpoint."
            )
        elif (
            config["auto_map"]["AutoConfig"] != "configuration_internvl_chat.InternVLChatConfig"
            or config["auto_map"]["AutoModel"] != "modeling_internvl_chat.InternVLChatModel"
            or config["auto_map"]["AutoModelForCausalLM"] != "modeling_internvl_chat.InternVLChatModel"
        ):
            logger.warning(
                f"AutoMapping in config.json in {path} does not match InternVLChat. "
                "Not a transformers compatible checkpoint."
            )
        else:
            ret["auto_map_ok"] = True
   
    # assert that the configuration.py and modeling.py files are present
    if (
        not osp.isfile(osp.join(path, 'configuration_internvl_chat.py'))
        or not osp.isfile(osp.join(path, 'modeling_internvl_chat.py'))
        or not osp.isfile(osp.join(path, 'configuration_intern_vit.py'))
        or not osp.isfile(osp.join(path, 'modeling_intern_vit.py'))
    ):
        logger.warning(
            f"configuration.py and/or modeling.py files are missing in {path}. "
            "Not a transformers compatible checkpoint."
        )
    else:
        ret["pyfiles_ok"] = True
    
    # check there is at least one safetensors file
    if not any(fname.endswith('.safetensors') for fname in os.listdir(path)):
        logger.warning(
            f"No .safetensors file found in {path}. "
            "Not a transformers compatible checkpoint."
        )
    else:
        ret["safetensors_ok"] = True
    
    if return_dict:
        return ret
    return all(ret.values())


def make_transformers_compatible(path: str) -> None:
    """
    Convert a checkpoint at path to be compatible with Hugging Face Transformers.
    This cannot be done if the checkpoint:
    - Does not have a config.json file
    - Does not have at least one .safetensors file
    A value error is raised in these cases.

    Otherwise, the following actions are taken if needed:
    - Ensuring the config.json file has the correct AutoMapping for InternVLChat
    - Ensuring the configuration_internvl_chat.py and modeling_internvl_chat.py files are present
    """
    ret = is_transformers_compatbile(path, return_dict=True)
    if all(ret.values()):
        return
    
    if not ret["config_ok"]:
        raise ValueError(
            f"Cannot convert checkpoint at {path} to transformers compatible format. "
            "No config.json found."
        )

    if not ret["safetensors_ok"]:
        raise ValueError(
            f"Cannot convert checkpoint at {path} to transformers compatible format. "
            "No .safetensors file found."
        )

    config_path = osp.join(path, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    if not ret["auto_map_ok"]:
        config['auto_map'] = {
            "AutoConfig": "configuration_internvl_chat.InternVLChatConfig",
            "AutoModel": "modeling_internvl_chat.InternVLChatModel",
            "AutoModelForCausalLM": "modeling_internvl_chat.InternVLChatModel"
        }
        logger.info(f"Added AutoMapping to config.json in {path}.")
        
        # maybe use a filelock here?
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    if not ret["pyfiles_ok"]:
        # copy the configuration and modeling files from the internvl package
        import internvl
        src_dir = osp.dirname(internvl.__file__)
        copy(
            osp.join(src_dir, 'model', 'internvl_chat', 'configuration_internvl_chat.py'),
            osp.join(path, 'configuration_internvl_chat.py')
        )
        copy(
            osp.join(src_dir, 'model', 'internvl_chat', 'modeling_internvl_chat.py'),
            osp.join(path, 'modeling_internvl_chat.py')
        )
        copy(
            osp.join(src_dir, 'model', 'internvl_chat', 'modeling_intern_vit.py'),
            osp.join(path, 'modeling_intern_vit.py')
        )
        copy(
            osp.join(src_dir, 'model', 'internvl_chat', 'configuration_intern_vit.py'),
            osp.join(path, 'configuration_intern_vit.py')
        )
        logger.info(f"Copied configuration_internvl_chat.py and modeling_internvl_chat.py to {path}.")
    
    assert is_transformers_compatbile(path), \
    f"Failed to convert checkpoint at {path} to transformers compatible format."
    
    logger.info(f"Checkpoint at {path} is now transformers compatible.")


def fix_checkpoint_config_mismatch(path: str, model: PreTrainedModel) -> None:
    """
    Fix a checkpoint at :path: whose config.json does not match the config of the :model:.
    This is useful when loading a checkpoint that was saved with a different
    version of the code, and the default model config.json file has changed.
    """
    config_path = osp.join(path, 'config.json')
    if not osp.isfile(config_path):
        raise ValueError(f"No config.json found in {path}. Cannot fix config mismatch.")
    
    config = PretrainedConfig.from_json_file(config_path).to_dict()
    correct_config = model.config.to_dict()
    
    # detect any updates
    updated = False
    for key, value in correct_config.items():
        if key not in config or config[key] != value:
            updated = True
            change = f"Updated key '{key}' to value '{value}'"
            logger.info(f"{change} in config.json in {path}.")
    
    # if updates detected, update the config with the model config
    # (uses the model.save_pretrained() interface to also dump the 
    # "architectures" field without manually patching)
    if updated:
        timestring = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        tmpdir = osp.join(path, f'tmp_{timestring}')
        os.makedirs(tmpdir, exist_ok=True)
        model.save_pretrained(tmpdir)
        copy(osp.join(tmpdir, 'config.json'), config_path)
        rmtree(tmpdir)
        logger.info(f"Updated config.json in {path} to match the correct configuration.")
    else:
        logger.info(f"Config.json in {path} already matches the correct configuration. No changes made.")