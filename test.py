import torch
"""
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.current_device())
"""

def get_index(lista,value):
    return [x for (x,m) in enumerate(lista) if m==value]

if __name__=="__main__":
    indexa=[1,2,3,2]
    result=get_index(indexa,3)
    print(result)



