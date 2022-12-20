import os
import matplotlib.pyplot as plt
import json
import argparse


def plot_metrics(
        logfile: str,
        outpath: str,
        ):
    with open(logfile) as f:
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
    parser.add_argument('--logfile', type=str)
    parser.add_argument('--outpath', type=str, default=os.getcwd())
    args = parser.parse_args()

    plot_metrics(logfile=args.logfile, outpath=args.outpath)

if __name__ == "__main__":
    main()
