  
import sys
 
def substring_search(file, substring):
    with open(file, 'r') as target_file:
        for _, line in enumerate(target_file.readlines(), 1):
                if substring in line:
                    res = line.replace('Value', '')
                    res = res.replace(' ', '')
                    res = res.replace('\n', '')
                    
        return res
            

if __name__ == '__main__':
    output_list = substring_search(sys.argv[1],sys.argv[2])
    sys.stdout.write(str(output_list))