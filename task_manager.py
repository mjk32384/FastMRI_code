from automation_json_func import auto
from email_me import email_me
import sys

tasks = [
    {'report_interval':500, 'epoch':10, 'batch_size':1, 'lr':1e-3,
     'net_name': 'test_6125_all_ssim_mask', 'cascade':6, 'chans':12, 'sens_chans':5,
     'acc_weight':{2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1},
     'previous_model':'', 'mask_mode':'equispaced','use_SSIM_mask_train':True},
    {'report_interval':500, 'epoch':10, 'batch_size':1, 'lr':1e-3,
     'net_name': 'test_6125_all_ssim_mask_random', 'cascade':6, 'chans':12, 'sens_chans':5,
     'acc_weight':{2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1},
     'previous_model':'', 'mask_mode':'random','use_SSIM_mask_train':True},
    {'report_interval':500, 'epoch':10, 'batch_size':1, 'lr':1e-3,
     'net_name': 'test_6125_all_ssim_mask_random_9', 'cascade':6, 'chans':12, 'sens_chans':5,
     'acc_weight':{2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:2, 10:1},
     'previous_model':'', 'mask_mode':'random','use_SSIM_mask_train':True},
    {'report_interval':500, 'epoch':10, 'batch_size':1, 'lr':1e-3,
     'net_name': 'test_6125_all_ssim_mask_random_910', 'cascade':6, 'chans':12, 'sens_chans':5,
     'acc_weight':{2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:2, 10:2},
     'previous_model':'', 'mask_mode':'random','use_SSIM_mask_train':True}
] #이것만 추가하고 돌리면 됨


email_string = ""
for i, task in enumerate(tasks):
    #Task Start
    print("Task %d (out of %d) started"%(i + 1, len(tasks)))
    if i != 0:
        email_string = email_string + "\nand also\n\n" 
    email_string = email_string +"Google Cloud\nTask %d (out of %d) started\n"%(i + 1, len(tasks))
    email_string = email_string + "========== Details ==========\n" + '\n'.join([key + " : " + str(value) for key, value in task.items()]) + "\n"
    email_me(title = "[Notice] Task %d (out of %d) Started"%(i + 1, len(tasks)), detail = email_string)

    #Process Task
    result = ""
    try:
        result = auto(task)
    except Exception as e:
        print("Task %d (out of %d) Terminated"%(i + 1, len(tasks)))
        print(e)
        email_me(title = "[Error] Task Terminated, Go Find Out in Google Cloud", detail = "Task %d (out of %d) Terminated"%(i + 1, len(tasks)) + "\nException : " + str(e) + "\n" + "Ooopss...")
        sys.exit(1)
        

    #End Task
    print("Task %d (out of %d) Completed"%(i + 1, len(tasks)))
    email_string = "Google Cloud\nTask %d (out of %d) completed\n"%(i + 1, len(tasks)) + '\n'.join(result.split('\n')[1:]) + "\n"
print("Done")
email_me(title = "[Notice] Google Cloud All Tasks Completed!", detail = email_string + "Congratulations!")
sys.exit(0)