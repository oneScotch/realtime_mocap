import os


def add_logger_path(process_config_dict: dict, log_dir: str):
    process_name = process_config_dict['name']
    log_file_path = os.path.join(log_dir, f'{process_name}_log.txt')
    logger_dict = process_config_dict['logger'] \
        if 'logger' in process_config_dict \
        else None
    if logger_dict is not None:
        logger_dict['logger_path'] = log_file_path
        process_config_dict['logger'] = logger_dict
    return process_config_dict


def add_logger_path_recursive(config_dict: dict, log_dir: str):
    for key, value in config_dict.items():
        if isinstance(value, dict):
            new_value = add_logger_path_recursive(
                config_dict=value, log_dir=log_dir)
            config_dict[key] = new_value
        else:
            continue
    if 'type' in config_dict and \
            'process' in config_dict['type'].lower():
        config_dict = add_logger_path(
            process_config_dict=config_dict, log_dir=log_dir)
    return config_dict


def add_logger_name(process_config_dict: dict, ):
    process_name = process_config_dict['name']
    logger_dict = process_config_dict['logger'] \
        if 'logger' in process_config_dict \
        else None
    if logger_dict is not None:
        logger_dict['logger_name'] = f'{process_name}_logger'
        process_config_dict['logger'] = logger_dict
    return process_config_dict


def add_logger_name_recursive(config_dict: dict):
    for key, value in config_dict.items():
        if isinstance(value, dict):
            new_value = add_logger_name_recursive(config_dict=value)
            config_dict[key] = new_value
        else:
            continue
    if 'type' in config_dict and \
            'process' in config_dict['type'].lower():
        config_dict = add_logger_name(process_config_dict=config_dict)
    return config_dict
