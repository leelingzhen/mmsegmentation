import os
import matplotlib.pyplot as plt
import json
import argparse


def find_latest_logfile(
        exp_dir:str
        ):
    json_logs= list()
    for f in os.listdir(exp_dir):
        if os.path.isfile(os.path.join(exp_dir, f)) and 'json' in f:
            json_logs.append(os.path.join(exp_dir, f))
    latest_log = max(json_logs, key=os.path.getctime)
    
    return latest_log


def plot_metrics(
        exp_dir:str,
        outpath: str,
        ):

    #getting latest log file from experiment
    latest_log_path = find_latest_logfile(exp_dir)

    with open(latest_log_path) as f:
        metrics = {}
        for i, json_object in enumerate(f):
            metric_object = json.loads(json_object)
            if i == 0:
                continue
            if i == 1:
                for key in metric_object.keys():
                    metrics[key] =  list()
            if metric_object['mode'] == 'train':
                for key in metric_object.keys():
                    metrics[key].append(metric_object[key])
    reject = ['iter', 'mode', 'epoch']
    figure, axis = plt.subplots(3,3)

    axis[0,0].plot(metrics['iter'], metrics['lr'])
    axis[0,0].set_title('lr')

    axis[0,1].plot(metrics['iter'], metrics['memory'])
    axis[0,1].set_title('memory')
    
    axis[1,0].plot(metrics['iter'], metrics['loss'])
    axis[1,0].set_title('loss')

    axis[1,1].plot(metrics['iter'], metrics['decode.loss_ce'])
    axis[1,1].set_title('decode.loss_ce')

    axis[1,2].plot(metrics['iter'], metrics['decode.acc_seg'])
    axis[1,2].set_title('decode.acc_seg')

    axis[2,0].plot(metrics['iter'], metrics['loss_val'])
    axis[2,0].set_title('loss_val')

    axis[2,1].plot(metrics['iter'], metrics['decode.loss_ce_val'])
    axis[2,1].set_title('decode.loss_ce_val')

    axis[2,2].plot(metrics['iter'], metrics['decode.acc_seg_val'])
    axis[2,2].set_title('decode.acc_seg_val')

    plt.show()



    return None

def main():
    parser = argparse.ArgumentParser(description='Render loss metrics of training')
    parser.add_argument('--exp_dir', type=str)
    parser.add_argument('--outpath', type=str, default=os.getcwd())
    args = parser.parse_args()

    plot_metrics(exp_dir=args.exp_dir, outpath=args.outpath)
    

if __name__ == "__main__":
    main()
