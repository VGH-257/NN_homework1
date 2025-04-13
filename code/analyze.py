import matplotlib.pyplot as plt
import numpy as np


def compare_lr(lr, dev_acc, test_acc):
    plt.figure(figsize=(8, 6))
    plt.plot(lr, dev_acc, marker='o', markersize=8, label='Dev Accuracy', color='#1f77b4', linewidth=2)
    plt.plot(lr, test_acc, marker='s', markersize=8, label='Test Accuracy', color='#ff7f0e', linewidth=2)

    plt.xscale('log')
    plt.xticks(lr, lr, fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlabel('Learning Rate', fontsize=30)
    plt.ylabel('Accuracy (%)', fontsize=30)
    # plt.title('Accuracy vs. Learning Rate', fontsize=14, pad=20)
    plt.legend(loc='best', fontsize=25)

    plt.grid(linestyle='--', alpha=0.6)
    plt.tight_layout()
    # plt.show()  
    plt.savefig('./figs/lr.png',dpi=300)  
    
    
    
def compare_hidden_size(hidden_size, dev_acc, test_acc):
    plt.figure(figsize=(8, 6))
    plt.plot(hidden_size, dev_acc, marker='o', markersize=8, label='Dev Accuracy', color='#1f77b4', linewidth=2)
    plt.plot(hidden_size, test_acc, marker='s', markersize=8, label='Test Accuracy', color='#ff7f0e', linewidth=2)

    plt.xticks(list(range(len(hidden_size))), hidden_size)
    plt.xticks(rotation=15,fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlabel('Hidden Size', fontsize=30)
    plt.ylabel('Accuracy (%)', fontsize=30)
    plt.legend(loc='best', fontsize=25)

    plt.grid(linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('./figs/hidden_size.png',dpi=300) 


def compare_L2(reg, dev_acc, test_acc):
    plt.figure(figsize=(8, 6))
    plt.plot(list(range(len(reg))), dev_acc, marker='o', markersize=8, label='Dev Accuracy', color='#1f77b4', linewidth=2)
    plt.plot(list(range(len(reg))), test_acc, marker='s', markersize=8, label='Test Accuracy', color='#ff7f0e', linewidth=2)

    plt.xticks(list(range(len(reg))), reg, fontsize=20)
    plt.yticks(fontsize=20)

    plt.xlabel('Reg Weight', fontsize=30)
    plt.ylabel('Accuracy (%)', fontsize=30)
    plt.legend(loc='best', fontsize=25)

    plt.grid(linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('./figs/reg.png',dpi=300) 


def compare_act_func(func, dev_acc, test_acc):
    plt.figure(figsize=(8, 6))
    bar_width = 0.35                          
    x = np.arange(len(func)) 

    rects1 = plt.bar(x - bar_width/2, dev_acc, width=bar_width, 
                    label='Dev Accuracy', color='#4e79a7', edgecolor='black')
    rects2 = plt.bar(x + bar_width/2, test_acc, width=bar_width, 
                    label='Test Accuracy', color='#f28e2b', edgecolor='black')

    plt.xlabel('Activation Function', fontsize=30)
    plt.ylabel('Accuracy (%)', fontsize=30)
    plt.xticks(x, func,fontsize=20)
    plt.yticks(fontsize=20)             
    plt.legend(loc='best', fontsize=25)  
    
    plt.grid(linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('./figs/act.png',dpi=300) 

if __name__=="__main__":
    lr = [0.001, 0.01, 0.1, 1]
    dev_acc = [x*100 for x in [0.1527,0.3088,0.4135,0.5279]]
    test_acc = [x*100 for x in [0.146,0.306,0.4283,0.5227]]
    compare_lr(lr, dev_acc, test_acc)
    
    hidden_size = ["[128, 64]","[256, 128]","[512, 256]","[1024, 512]"]
    dev_acc = [x*100 for x in [0.5139,0.5168,0.5205,0.5279]]
    test_acc = [x*100 for x in [0.5004,0.5132,0.515,0.5227]]
    compare_hidden_size(hidden_size, dev_acc, test_acc)
    
    reg=[0,0.001,0.01,0.1]
    dev_acc = [x*100 for x in[0.5279, 0.3775,0.1,0.0963]]
    test_acc = [x*100 for x in[0.5227, 0.3825, 0.1, 0.1]]
    compare_L2(reg, dev_acc, test_acc)
    
    fuct = ['Relu','Sigmoid','Tanh']
    dev_acc = [x*100 for x in[0.5279,0.3482,0.4568]]
    test_acc = [x*100 for x in[0.5227,0.322,0.2853]]
    compare_act_func(fuct, dev_acc, test_acc)
