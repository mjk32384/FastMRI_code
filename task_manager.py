from automation_json_func import auto
tasks = [
    {'report_interval':100, 'epoch':30, 'batch_size':1, 'lr':1e-3,
     'net_name': 'test_6125_59_1323', 'cascade':6, 'chans':12, 'sens_chans':5, 'acc_weight':{4:1}, 'previous_model':'test_6125_59_1323'}, #추가 학습 to epoch 30
    {'report_interval':100, 'epoch':10, 'batch_size':1, 'lr':1e-3,
     'net_name': 'test_6125_59_1323', 'cascade':6, 'chans':12, 'sens_chans':5,
     'acc_weight':{2:1/9, 3:1/9, 4:1/9, 5:1/9, 6:1/9, 7:1/9, 8:1/9, 9:1/9, 10:1/9}, 'previous_model':'test_6125_59_1323'}
]

for i, task in enumerate(tasks):
    print("task (%d/%d) started"%(i, len(tasks))
    auto(task)
    print("task (%d/%d) complete"%(i, len(tasks)))