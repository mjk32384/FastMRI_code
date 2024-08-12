from automation_json_func import auto
from email_me import email_me
import sys

tasks = [
    {'report_interval':500, 'epoch':30, 'batch_size':1, 'lr':1e-3,
     'net_name': 'test_6125_59_ver3', 'cascade':6, 'chans':12, 'sens_chans':5, 'acc_weight':{4:1}, 'previous_model':'test_6125_59_ver2'}, #추가 학습 to epoch 30
    {'report_interval':500, 'epoch':10, 'batch_size':1, 'lr':1e-3,
     'net_name': 'test_6125_all', 'cascade':6, 'chans':12, 'sens_chans':5,
     'acc_weight':{2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1}, 'previous_model':''} #23456789 균등분배
] #이것만 추가하고 돌리면 됨


email_string = ""
for i, task in enumerate(tasks):
    #Task Start
    print("Task %d (out of %d) started"%(i + 1, len(tasks)))
    if i != 0:
        email_string = email_string + "\nand also\n\n" 
    email_string = email_string +"Vessl1\nTask %d (out of %d) started\n"%(i + 1, len(tasks))
    email_string = email_string + "========== Details ==========\n" + '\n'.join([key + " : " + str(value) for key, value in task.items()]) + "\n"
    email_me(title = "[Notice] Task %d (out of %d) Started"%(i + 1, len(tasks)), detail = email_string)

    #Process Task
    result = ""
    try:
        result = auto(task)
    except Exception as e:
        print("Task %d (out of %d) Terminated"%(i + 1, len(tasks)))
        print(e)
        email_me(title = "[Error] Task Terminated, Go Find Out in Vessl 1", detail = "Task %d (out of %d) Terminated"%(i + 1, len(tasks)) + "\nException : " + str(e) + "\n" + "Ooopss...")
        sys.exit(1)
        

    #End Task
    print("Task %d (out of %d) Completed"%(i + 1, len(tasks)))
    email_string = "Vessl1\nTask %d (out of %d) completed\n"%(i + 1, len(tasks)) + '\n'.join(result.split('\n')[1:]) + "\n"
print("Done")
email_me(title = "[Notice] Vessl1 All Tasks Completed!", detail = "Congratulations!")
sys.exit(0)