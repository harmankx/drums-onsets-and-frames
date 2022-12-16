import os
from datetime import datetime

import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from evaluate import evaluate
from onsets_and_frames import *
from random import randrange
import pandas as pd

ex = Experiment('train_transcriber')

@ex.config
def config():
    logdir = 'runs/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iterations = 150000
    resume_iteration = None
    checkpoint_interval = 1000
    train_on = 'GROOVE'

    batch_size = 8
    sequence_length = SEQUENCE_LENGTH 
    model_complexity = 16

    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory < 10e9:
        batch_size //= 2
        sequence_length //= 2
        print(f'Reducing batch size to {batch_size} and sequence_length to {sequence_length} to save memory')
    learning_rate = 0.0006
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98

    leave_one_out = None

    clip_gradient_norm = 3

    validation_length = sequence_length
    validation_interval = 1000

    ex.observers.append(FileStorageObserver.create(logdir))


@ex.automain
def train(logdir, device, iterations, resume_iteration, checkpoint_interval, train_on, batch_size, sequence_length,
          model_complexity, learning_rate, learning_rate_decay_steps, learning_rate_decay_rate, leave_one_out,
          clip_gradient_norm, validation_length, validation_interval):
    print_config(ex.current_run)

    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    dataset = GROOVE(groups=['train_CE_normalized15db'], sequence_length=sequence_length)
    train_eval = GROOVE(groups=['train_CE_normalized15db'], sequence_length=sequence_length)
    validation_dataset = GROOVE(groups=['validation_CE_normalized15db'], sequence_length=validation_length)

    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

    if resume_iteration is None:
        model = OnsetsAndFrames(N_MELS, 8, model_complexity).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'model-{resume_iteration}.pt')
        model = torch.load(model_path)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))

    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    val_losses = []
    val_vel_losses = []
    val_onset_losses = []
    val_iter = []
    val_acc = []

    train_losses = []
    train_vel_losses = []
    train_onset_losses = []
    train_iter = []
    train_acc = []

    #   getting accuracies to a file to evaluate
    onset_f1_dict = {}
    velocity_f1_dict = {}

    onset_loss_dict = {}
    velocity_loss_dict = {}

    for i, batch in zip(loop, cycle(loader)):
        predictions, losses = model.run_on_batch(batch)

        loss = sum(losses.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if clip_gradient_norm:
            clip_grad_norm_(model.parameters(), clip_gradient_norm)

        for key, value in {'loss': loss, **losses}.items():
            writer.add_scalar(key, value.item(), global_step=i)
        
        if i % validation_interval == 0:
            val_losses_num = 0
            val_vel = 0
            val_onset = 0
            model.eval()
            with torch.no_grad():
                eval_ = evaluate(validation_dataset, model)
                if not onset_f1_dict:
                    for item in eval_['path']:
                        onset_f1_dict[item] = []
                        onset_loss_dict[item] = []
                        velocity_loss_dict[item] = []
                        velocity_f1_dict[item] = []
                else:
                    for idx, item in enumerate(eval_['path']):
                        onset_f1_dict[item].append(eval_['metric/total/f1'][idx])
                        velocity_f1_dict[item].append(eval_['metric/total-with-velocity/f1'][idx])

                        onset_loss_dict[item].append(eval_['loss/onset'][idx])
                        velocity_loss_dict[item].append(eval_['loss/velocity'][idx])

                total_velocity = np.mean(eval_['metric/total-with-velocity/f1'])
                loss_onset = np.mean(eval_['loss/onset'])
                loss_velocity = np.mean(eval_['loss/velocity'])

                val_onset += loss_onset
                val_vel += loss_velocity
                val_losses_num += loss_onset + loss_velocity
                val_acc.append(total_velocity)

                for key, value in eval_.items():
                    if(key != 'path' and key != 'metric/total-with-velocity/loss'):
                        writer.add_scalar('validation/' + key.replace(' ', '_'), np.mean(value), global_step=i)
            
            val_losses.append(val_losses_num)
            val_vel_losses.append(val_vel)
            val_onset_losses.append(val_onset)
            val_iter.append(i) 
            model.train()

        if i % checkpoint_interval == 0:
            train_losses.append(loss.item())
            train_vel_losses.append(losses['loss/velocity'].item())
            train_onset_losses.append(losses['loss/onset'].item())
            train_iter.append(i)
            acc = evaluate(train_eval, model)["metric/total-with-velocity/f1"]
            train_acc.append(np.mean(acc))

            torch.save(model, os.path.join(logdir, f'model-{i}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))


            path = os.path.join(logdir, f'plots-{i}')
            if not os.path.isdir(path):
                os.makedirs(path)

            # total loss
            plt.plot(train_iter, train_losses, label = "Train Loss")
            plt.plot(val_iter, val_losses, label = "Val Loss")
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            # plt.yscale('log')
            plt.legend()
            plt.title("Total Loss")
            plt.savefig(os.path.join(path, 'total-loss.png'), format='png')
            # plt.show()
            plt.close()

            # vel loss
            plt.plot(train_iter, train_vel_losses, label = "Train Velocity Loss")
            plt.plot(val_iter, val_vel_losses, label = "Val Velocity Loss")
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            # plt.yscale('log')
            plt.legend()
            plt.title("Velocity Loss")
            plt.savefig(os.path.join(path, 'velocity-loss.png'), format='png')
            # plt.show()
            plt.close()

            # onset loss
            plt.plot(train_iter, train_onset_losses, label = "Train Onset Loss")
            plt.plot(val_iter, val_onset_losses, label = "Val Onset Loss")
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            # plt.yscale('log')
            plt.legend()
            plt.title("Onset Loss")
            plt.savefig(os.path.join(path, 'onset-loss.png'), format='png')
            # plt.show()
            plt.close()

            # accuracy plot
            plt.plot(train_iter, train_acc, label = "Train Accuracy")
            plt.plot(val_iter, val_acc, label = "Val Accuracy")
            plt.xlabel('Iterations')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(os.path.join(path, 'accuracy.png'), format='png')
            # plt.show()
            plt.close()

    path = os.path.join(logdir, 'accuracies')
    if not os.path.isdir(path):
        os.makedirs(path)
    
    onset_df = pd.DataFrame.from_dict(onset_f1_dict, orient='index')
    velocity_df = pd.DataFrame.from_dict(velocity_f1_dict, orient='index')
    onset_loss_df = pd.DataFrame.from_dict(onset_loss_dict, orient='index')
    velocity_loss_df = pd.DataFrame.from_dict(velocity_loss_dict, orient='index')

    onset_df.to_csv(os.path.join(path, 'onset_f1.csv'))
    onset_loss_df.to_csv(os.path.join(path, 'onset_loss.csv'))
    velocity_df.to_csv(os.path.join(path, 'velocity_df.csv'))
    velocity_loss_df.to_csv(os.path.join(path, 'velocity_loss.csv'))



    path = os.path.join(logdir, 'plots')
    if not os.path.isdir(path):
        os.makedirs(path)

    # total loss
    plt.plot(train_iter, train_losses, label = "Train Loss")
    plt.plot(val_iter, val_losses, label = "Val Loss")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.title("Total Loss")
    plt.savefig(os.path.join(logdir, 'plots', 'total-loss.png'), format='png')
    # plt.show()
    plt.close()

    # vel loss
    plt.plot(train_iter, train_vel_losses, label = "Train Velocity Loss")
    plt.plot(val_iter, val_vel_losses, label = "Val Velocity Loss")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.title("Velocity Loss")
    plt.savefig(os.path.join(logdir, 'plots', 'velocity-loss.png'), format='png')
    # plt.show()
    plt.close()

    # onset loss
    plt.plot(train_iter, train_onset_losses, label = "Train Onset Loss")
    plt.plot(val_iter, val_onset_losses, label = "Val Onset Loss")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.title("Onset Loss")
    plt.savefig(os.path.join(logdir, 'plots', 'onset-loss.png'), format='png')
    # plt.show()
    plt.close()

    # accuracy plot
    plt.plot(train_iter, train_acc, label = "Train Accuracy")
    plt.plot(val_iter, val_acc, label = "Val Accuracy")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(logdir, 'plots', 'accuracy.png'), format='png')
    # plt.show()
    plt.close()