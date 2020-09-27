

def clip_max_length(train_data, data_bundle, max_sent_len=50):
    """
    将数据中超过max_len的数据截取到50这个长度

    :param train_data:
    :param data_bundle:
    :param max_sent_len:
    :return:
    """
    train_data.apply_field(lambda x: x[:max_sent_len] + ([] if len(x) < max_sent_len else [x[-1]]),
                   field_name='input', new_field_name='input')
    if train_data.has_field('language_ids'):
        train_data.apply_field(lambda x: x[:max_sent_len] + ([] if len(x) < max_sent_len else [x[-1]]),
                       field_name='language_ids', new_field_name='language_ids')
    train_data.add_seq_len('input')

    # train_data.drop(lambda ins:ins['seq_len']>50, inplace=True)
    for name, ds in data_bundle.iter_datasets():
        if 'train' in name:
            continue
        numbers = len(list(filter(lambda x: x > max_sent_len, map(len, ds.get_field('input').content))))
        print(f'{name}:{numbers}')
        ds.apply_field(lambda x:x[:max_sent_len] + ([] if len(x)<max_sent_len else [x[-1]]),
                       field_name='input', new_field_name='input')
        if ds.has_field('language_ids'):
            ds.apply_field(lambda x:x[:max_sent_len] + ([] if len(x)<max_sent_len else [x[-1]]),
                           field_name='language_ids', new_field_name='language_ids')
