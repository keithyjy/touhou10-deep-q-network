import psutil

# Finds and returns the process ID of a given process name
# Credits to this post: https://stackoverflow.com/questions/43009612/is-there-a-way-to-check-if-a-specific-program-is-running-using-python
def find_process(proc_name):
        for proc in psutil.process_iter():
            try:
                pinfo = proc.as_dict(attrs=['pid', 'name'])
                if proc_name.lower() in pinfo['name'].lower():
                    return pinfo['pid']
            except:
                return -1
 
