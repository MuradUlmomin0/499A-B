"""
This is training endpoint.
@author
- Van Tuan Nguyen (vantuan.nguyen@lqdtu.edu.vn)
- Razvan Beuran (razvan@jaist.ac.jp)
@create date 2023-12-11 00:28:29
"""

import os  # file/folder path handle করে
import json  # JSON file পড়ে
import pickle     # latent data save করে
import pandas as pd  # table/data handle করে
import numpy as np  # numerical/array calculation
import torch  # neural network training
import argparse  # command-line argument handle করে
import copy  # model weight copy করে
import random  # random client select করে
import json  # JSON file পড়ে
from torch.utils.data import DataLoader, random_split, ConcatDataset  # data batch/split/combine করে
from Model import Shrink_Autoencoder  # main SAE model
from Model import Autoencoder  # normal Autoencoder model
from DataLoader import load_data  # data load করে
from DataLoader import IoTDataset  # PyTorch dataset বানায়
from DataLoader import IoTDataProccessor  # data process/scale করে
from Trainer import ClientTrainer  # local training করে
from Trainer import GlobalAggregator  # local models combine করে
from Evaluator import Evaluator  # model test করে

import logging  # training message দেখায়

# Configure the logging module
logging.basicConfig(level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)  # logging setup করে
                    format='%(asctime)s - %(levelname)s - %(message)s')  # log format সেট করে


num_participants = 0.5  # 50% client participate করে
epoch = 5  # client 5 epoch train করে
num_rounds = 3  # FL 3 round চলে
lr_rate = 1e-5  # learning rate
shrink_lambda = 10  # SAE shrink parameter
network_size = 10  # total client = 10
data_seed = 1234  # random seed
# no_Exp = f"nonIID_Exp1_Rerun_{epoch}epoch_10client_lr0001_lamda{shrink_lambda}_ratio{num_participants*100}"
no_Exp = f"IID-Update_Exp6_scale_{epoch}epoch_{network_size}client_{num_rounds}rounds_lr{lr_rate}_lamda{shrink_lambda}_ratio{num_participants*100}_dataseed{data_seed}"  # experiment name বানায়

num_runs = 1  # experiment 1 বার চলে
batch_size = 12  # এক batch-এ 12 sample

new_device = True  # new normal test data নেয়
min_val_loss = float("inf")  # best loss শুরুতে infinite
global_patience = 1  # early stopping patience
global_worse = 0  # counter reset
metric = "AUC" #AUC or classification  # AUC দিয়ে performance মাপে
# model_type = "autoencoder"   #autoencoder; hybrid;
# update_type = "mse_avg"  #avg; fusion_avg; mse_avg
dim_features = 115   #nba-iot: 115; cic-2023: 46  # N-BaIoT-এর 115 feature

scen_name = 'FL-IoT'   # scenario name


config_file = "Configuration/scen2-nba-iot-10clients.json"  # 10-client config file
# config_file = "Configuration/scen2-nba-iot-10clients.json"
# config_file = "Configuration/cic-config.json"

def set_seeds(seed):  # random seed function
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch random seed
    if torch.cuda.is_available():  # GPU আছে কিনা দেখে
        torch.cuda.manual_seed_all(seed)  # GPU seed set করে

if __name__ == "__main__":  # file run হলে main কাজ শুরু
        random.seed(data_seed)  # main random seed set
        np.random.seed(data_seed)  # NumPy seed set
        try:  # error ছাড়া run করার চেষ্টা
            logging.info("Loading configuration...")  # config load message
            with open(config_file, "r") as config_file:  # config file খোলে
                config = json.load(config_file)  # JSON config পড়ে
        except Exception as e:  # error হলে এখানে আসে
            logging.info("Failed to load configuration.")  # config fail message
        
        devices_list = random.sample(config['devices_list'], network_size)  # 10 client select করে
        # devices_list = config['devices_list']
        client_info = []  # client info রাখার list
        # random.seed(data_seed)
        # np.random.seed(data_seed)
        for device in devices_list:  # প্রতি client নিয়ে loop
            logging.info("Creating metadata for client...")  # client setup message
            normal_data_path = os.path.join(config['data_path'], device["normal_data_path"])  # normal data path
            abnormal_data_path = os.path.join(config['data_path'], device["abnormal_data_path"])  # attack data path
            test_new_normal_data_path = os.path.join(config['data_path'], device["test_normal_data_path"])  # test-normal path
            
            logging.info("Loading data from {}...".format(device['name']))  # client data load message
            
            # normal_data = load_data(normal_data_path, header="infer")
            normal_data = load_data(normal_data_path)  # normal data load করে
            normal_data = normal_data.sample(frac=1).reset_index(drop=True)  # normal data shuffle করে
            # abnormal_data = load_data(abnormal_data_path, header="infer")

            abnormal_data = load_data(abnormal_data_path)  # attack data load করে
            abnormal_data = abnormal_data.sample(frac=1).reset_index(drop=True)  # attack data shuffle করে
            
            # new normal data from new devices
            if new_device:  # new normal data থাকলে
                new_normal_data = load_data(test_new_normal_data_path)  # new normal data load করে
            
            device_name = device['name']  # client name নেয়
            print(f"{device_name} has {len(normal_data)} normal data and {len(abnormal_data)} abnormal data")  # data count দেখায়
            # now, need to split data before normalization
            train_normal_size = int(0.4 * len(normal_data))  # 40% train
            valid_normal_size = int(0.1 * len(normal_data))  # 10% validation
            dev_normal_size = int(0.4 * len(normal_data))  # 40% development
            test_normal_size = len(normal_data) - train_normal_size - valid_normal_size - dev_normal_size  # বাকি data test
            
            train_normal_data = normal_data[:train_normal_size]  # train data নেয়
            valid_normal_data = normal_data[train_normal_size:train_normal_size+valid_normal_size]  # validation data নেয়
            dev_normal_data = normal_data[train_normal_size+valid_normal_size:train_normal_size+valid_normal_size+dev_normal_size]  # development data নেয়
            test_normal_data = normal_data[train_normal_size+valid_normal_size+dev_normal_size:]  # test normal data নেয়

            data_processor = IoTDataProccessor(scaler="standard")  # Standard Scaling করে
            processed_train_data, train_label = data_processor.fit_transform(train_normal_data)  # train data process করে
            processed_valid_data, valid_label = data_processor.transform(valid_normal_data)  # validation data process করে
            # processed_dev_data, dev_label = data_processor.transform(dev_normal_data)
            processed_test_data, test_label = data_processor.transform(test_normal_data)  # test data process করে
            processed_abnormal_data, abnormal_label = data_processor.transform(abnormal_data, type="abnormal")  # attack data process করে
            
            if new_device:  # new normal data থাকলে
                processed_new_normal_data, new_normal_label = data_processor.transform(new_normal_data)  # new normal data process করে
                processed_test_data = np.concatenate([processed_test_data, processed_new_normal_data], axis=0)  # normal test data combine করে
                processed_test_label = np.concatenate([test_label, new_normal_label], axis=0)  # test labels combine করে
                test_dataset = IoTDataset(processed_test_data, processed_test_label)  # normal test dataset বানায়
            else:  # if false হলে চলে
                test_dataset = IoTDataset(processed_test_data, test_label)  # test dataset বানায়
            
            train_dataset = IoTDataset(processed_train_data, train_label)  # train dataset বানায়
            valid_dataset = IoTDataset(processed_valid_data, valid_label)  # validation dataset বানায়
            # dev_dataset = IoTDataset(processed_dev_data, dev_label)
            
            
            # indices = np.random.choice(processed_abnormal_data.shape[0], 3000, replace=False)
            # unique_values, counts = np.unique(abnormal_label[indices], return_counts=True)
            # print(f"Abnormal data: {unique_values} - {counts}")
            # abnormal_dataset = IoTDataset(processed_abnormal_data[indices], abnormal_label[indices])
            abnormal_dataset = IoTDataset(processed_abnormal_data, abnormal_label)  # attack dataset বানায়
            
            test_dataset = ConcatDataset([test_dataset, abnormal_dataset])  # normal + attack test combine করে

            train_loader = DataLoader(  # train DataLoader শুরু
                dataset=train_dataset,  # train dataset দেয়
                batch_size=batch_size,  # batch size 12
                pin_memory=True  # memory loading option
            )  # উপরের block শেষ
            valid_loader = DataLoader(  # validation DataLoader শুরু
                dataset=valid_dataset,  # validation dataset দেয়
                batch_size=batch_size,  # batch size 12
                pin_memory=True  # memory loading option
            )  # উপরের block শেষ
            test_loader = DataLoader(  # test DataLoader শুরু
                dataset=test_dataset,  # test dataset দেয়
                batch_size=batch_size,  # batch size 12
                pin_memory=True  # memory loading option
            )  # উপরের block শেষ
            
            # indices = np.random.choice(processed_dev_data.shape[0], 200, replace=False)
            client_info.append({  # client info save শুরু
                "device": device['name'],  # client name save
                "save_dir": "",  # save path empty
                "train_loader": train_loader,  # train loader save
                "valid_loader": valid_loader,  # validation loader save
                "test_loader": test_loader,  # test loader save
                "test_dataset": (processed_test_data, test_label),  # test data save
                "dev_normal_dataset": dev_normal_data  # development data save
            })  # উপরের block শেষ
        for update_type in ["mse_avg"]:  # MSEAvg aggregation ব্যবহার করে
        # for update_type in ["fedprox"]:
        # for update_type in ["mse_avg"]:
            # for model_type in ["autoencoder"]:
            for model_type in ["hybrid"]:  # Hybrid SAE-CEN ব্যবহার করে
                for run in range(num_runs):  # experiment run loop
                    set_seeds(run*10000)  # run seed set করে
                    for client in client_info:  # সব client নিয়ে loop
                        client['save_dir'] = os.path.join(f"Checkpoint/{network_size}/{no_Exp}/{run}/ClientModel", scen_name, model_type, update_type, client['device'])  # client model save path বানায়
                    global_worse = 0  # counter reset
                    min_val_loss = float("inf")  # best loss শুরুতে infinite
                    if True:  # এই block সবসময় চলে
                        # random.seed(run*10000)
                        
                        # devices_list = config['devices_list']

                        directory = f'Checkpoint/Results/Update/{network_size}/{no_Exp}/Run_{run}/{metric}'  # result folder path বানায়
                        if not os.path.exists(directory):  # folder আছে কিনা দেখে
                            os.makedirs(directory)  # result folder বানায়

                        # Check if the file exists and delete its content if it does
                        filename = f'{directory}/{scen_name}_{num_participants}_{model_type}_{update_type}_results.json'  # result filename বানায়
                        open(filename, 'w').close()  # পুরোনো result clear করে
                        
                        if model_type == "hybrid":  # hybrid model block
                            global_model = Shrink_Autoencoder(input_dim=dim_features,  # Shrink Autoencoder বানায়
                                                                output_dim=dim_features,  # output = 115 feature
                                                                shrink_lambda=shrink_lambda,  # shrink lambda = 10
                                                                latent_dim=11,  # latent dimension = 11
                                                                hidden_neus=50)  # hidden neuron = 50
                            
                            global_aggregator = GlobalAggregator(global_model, update_type=update_type)  # MSEAvg aggregator বানায়
                            
                            # Calculate the minimum length of all clients' datasets
                            min_len = min([len(client['dev_normal_dataset']) for client in client_info])  # smallest dev size নেয়

                            # Sample min_len data points from each client's dataset and create dev_dataset
                            dev_dataset = []  # dev data list
                            for client in client_info:  # সব client নিয়ে loop
                                sample_data = client['dev_normal_dataset'].sample(n=min_len)  # dev sample নেয়
                                dev_dataset.append(sample_data)  # sample list-এ রাখে
                                # client['dev_normal_dataset'] = client['dev_normal_dataset'].drop(sample_data.index)

                            # Concatenate all the sampled data into a single numpy array
                            dev_dataset = np.concatenate(dev_dataset, axis=0)  # সব dev data combine করে

                            global_aggregator.create_dev_dataset({"dataset": dev_dataset})  # dev data aggregator-কে দেয়
                            
                            # Now all clients' datasets have the same size
                            
                            # indices = np.random.choice(processed_dev_data.shape[0], 200, replace=False)
                            # dev_dataset = np.concatenate([client['dev_normal_dataset'][0] for client in client_info], axis=0)
                            # dev_label = np.concatenate([client['dev_normal_dataset'][1] for client in client_info], axis=0)
                            # global_aggregator.create_dev_dataset({"dataset": dev_dataset, "label": dev_label})
                            
                            # dev_dataset = np.concatenate([client['dev_normal_dataset'][0][indices] for client in client_info], axis=0)
                            # dev_label = np.concatenate([client['dev_normal_dataset'][1][indices] for client in client_info], axis=0)
                            # global_aggregator.create_dev_dataset({"dataset": dev_dataset, "label": dev_label})
                        
                            # global_test_data = np.concatenate([client['test_dataset'][0] for client in client_info], axis=0)
                            # global_test_label = np.concatenate([client['test_dataset'][1] for client in client_info], axis=0)
                            # global_test_dataset = IoTDataset(global_test_data, global_test_label)
                            # global_test_dataloader = DataLoader(
                            #     dataset=global_test_dataset,
                            #     batch_size=batch_size,
                            #     pin_memory=True
                            # )
                            
                            # Start training process
                            results = []  # result list
                            client_latent = {}  # latent result dictionary
                            for round in range(num_rounds):  # 3 federated round চালায়
                                client_latent[round] = {}  # round latent space
                                dev_dataset = []  # dev data list
                                dev_label = []  # dev label list
                                selected_idx = random.sample([i for i in range(len(client_info))], int(num_participants*len(client_info)))  # 10 থেকে 5 client select করে
                                selected_clients = [client_info[i] for i in selected_idx]  # selected client list
                                
                                total_training_samples = sum([len(client['train_loader'].dataset) for client in selected_clients])  # মোট training sample count
                                
                                # for client in client_info:
                                #     # indices = np.random.choice(client['dev_normal_dataset'].shape[0], 50, replace=False)
                                #     n_samples = min(20, len(client['dev_normal_dataset']))
                                #     sample_data = client['dev_normal_dataset'].sample(n=n_samples)
                                #     dev_dataset.append(sample_data)
                                #     client['dev_normal_dataset'] = client['dev_normal_dataset'].drop(sample_data.index)

                                # dev_dataset = np.concatenate(dev_dataset, axis=0)
                                # dev_label = np.concatenate(dev_label, axis=0)
                                # dev_dataset = np.concatenate([client['dev_normal_dataset'] for client in client_info], axis=0)
                                # global_aggregator.create_dev_dataset({"dataset": dev_dataset, "label": dev_label})
                                # global_aggregator.create_dev_dataset({"dataset": dev_dataset})
                                # Choose clients to train
                                # random.seed(round*1234)
                                # num_participants = random.uniform(0,1)
                                
                                client_weights = []  # local weights list
                                # if round == 0:
                                for i, client in enumerate(selected_clients):  # selected client train loop
                                    logging.info("Training local model...")  # local training message
                                    # ClientTrainer বানায়
                                    device_trainer = ClientTrainer(model=global_aggregator.model, \
                                        save_dir=client['save_dir'], epoch=epoch, lr_rate=lr_rate, update_type=update_type)  # trainer settings দেয়
                                    
                                    device_trainer.run(client["train_loader"], client["valid_loader"])  # local model train করে
                                    client_weights.append((copy.deepcopy(device_trainer.model.state_dict()), total_training_samples, len(client["train_loader"].dataset)))  # trained weights collect করে
                                    logging.info(f"Client {i} training done!")  # client training done message
                                    
                                # client_weights = random.sample(client_weights, int(num_participants * len(client_weights)))
                                global_aggregator.update(local_models=client_weights)  # MSEAvg দিয়ে global model বানায়
                                os.makedirs("../Outputs", exist_ok=True)  # Outputs folder বানায়
                                torch.save(global_aggregator.model.state_dict(), "../Outputs/global_model.pt")  # global model save করে

                                # global model update message
                                logging.info(f"Round {round+1}/{num_rounds} - Updated global model - \
                                    Global loss: {global_aggregator.val_loss}")  # global loss দেখায়
                                
                                logging.info("Training done! Evaluating...")  # evaluation শুরু
                                # evaluate the model in clients
                            
                                evaluator = Evaluator(global_aggregator.model, metric=metric, model_type=model_type)  # Evaluator বানায়
                                round_results = {}  # round result রাখে
                                
                                for i, client in enumerate(client_info):  # সব client evaluate করে
                                    logging.info(f"Evaluating client {i} - name: {client['device']}")  # client evaluation message
                                    auc_score, test_latent, test_label = evaluator.evaluate(client["test_loader"], client["train_loader"])  # AUC বের করে
                                    round_results[client['device']] = auc_score  # client AUC save করে
                                    # store latent of SAE and SAE_MSEFed
                                    client_latent[round][client['device']] = (test_latent, test_label)  # latent data save করে
                                round_results["global_loss"] = global_aggregator.val_loss  # global loss save করে
                                round_results['join_clients'] = selected_idx  # joined clients save করে
                                round_results = {f'round_{round+1}': round_results}  # round result গুছায়
                                
                                # Append to the JSON file
                                with open(filename, 'a') as f:  # result file খোলে
                                    f.write(json.dumps(round_results) + '\n')  # result JSON-এ লেখে
                                
                                if global_aggregator.val_loss < min_val_loss:  # loss কমেছে কিনা দেখে
                                    min_val_loss = global_aggregator.val_loss  # best loss update করে
                                    global_worse = 0  # counter reset
                                
                                if global_aggregator.val_loss >= min_val_loss:  # loss improve হয়নি কিনা দেখে
                                    global_worse += 1  # counter বাড়ায়
                                    if global_worse > global_patience:  # patience শেষ কিনা দেখে
                                        logging.info("Early stopping in global round!")  # early stop message
                                        break  # training loop বন্ধ করে
                            # store latent data of SAE and SAE_MSEFed for all rounds
                            # Define the file path
                            file_path = f'Checkpoint/LatentData/{network_size}/{no_Exp}/Run_{run}/latent_{model_type}_{update_type}.pkl'  # latent file path বানায়

                            # Create the directory if it does not exist
                            os.makedirs(os.path.dirname(file_path), exist_ok=True)  # latent folder বানায়

                            # Now you can safely write the file
                            with open(file_path, 'wb') as f:  # pickle file খোলে
                                pickle.dump(client_latent, f)  # latent data save করে
                            
                        if model_type == "autoencoder":  # Autoencoder branch; এখন run হয় না
                            global_model = Autoencoder(input_dim=dim_features,  # Autoencoder বানায়
                                                                output_dim=dim_features,  # output = 115 feature
                                                                latent_dim=11,  # latent dimension = 11
                                                                hidden_neus=50)  # hidden neuron = 50
                            
                            global_aggregator = GlobalAggregator(global_model, update_type=update_type)  # MSEAvg aggregator বানায়
                            # Calculate the minimum length of all clients' datasets
                            min_len = min([len(client['dev_normal_dataset']) for client in client_info])  # smallest dev size নেয়

                            # Sample min_len data points from each client's dataset and create dev_dataset
                            dev_dataset = []  # dev data list
                            for client in client_info:  # সব client নিয়ে loop
                                sample_data = client['dev_normal_dataset'].sample(n=min_len)  # dev sample নেয়
                                dev_dataset.append(sample_data)  # sample list-এ রাখে
                                # client['dev_normal_dataset'] = client['dev_normal_dataset'].drop(sample_data.index)

                            # Concatenate all the sampled data into a single numpy array
                            dev_dataset = np.concatenate(dev_dataset, axis=0)  # সব dev data combine করে

                            global_aggregator.create_dev_dataset({"dataset": dev_dataset})  # dev data aggregator-কে দেয়
                            
                            # dev_dataset = np.concatenate([client['dev_normal_dataset'][0] for client in client_info], axis=0)
                            # dev_label = np.concatenate([client['dev_normal_dataset'][1] for client in client_info], axis=0)
                            # global_aggregator.create_dev_dataset({"dataset": dev_dataset, "label": dev_label})
                            
                            
                            
                            # global_test_data = np.concatenate([client['test_dataset'][0] for client in client_info], axis=0)
                            # global_test_label = np.concatenate([client['test_dataset'][1] for client in client_info], axis=0)
                            # global_test_dataset = IoTDataset(global_test_data, global_test_label)
                            # global_test_dataloader = DataLoader(
                            #     dataset=global_test_dataset,
                            #     batch_size=batch_size,
                            #     pin_memory=True
                            # )
                            
                            # Start training process
                            results = []  # result list
                            for round in range(num_rounds):  # 3 federated round চালায়
                                dev_dataset = []  # dev data list
                                dev_label = []  # dev label list
                                dev_dataset = []  # dev data list
                                dev_label = []  # dev label list
                                # selected_idx = random.sample([i for i in range(len(client_info))], int(num_participants*len(client_info)))
                                # selected_clients = [client_info[i] for i in selected_idx]
                                # for client in client_info:
                                #     # indices = np.random.choice(client['dev_normal_dataset'].shape[0], 50, replace=False)
                                #     n_samples = min(20, len(client['dev_normal_dataset']))
                                #     sample_data = client['dev_normal_dataset'].sample(n=n_samples)
                                #     dev_dataset.append(sample_data)
                                #     client['dev_normal_dataset'] = client['dev_normal_dataset'].drop(sample_data.index)
                                
                                # dev_dataset = np.concatenate(dev_dataset, axis=0)
                                # dev_label = np.concatenate(dev_label, axis=0)
                                # dev_dataset = np.concatenate([client['dev_normal_dataset'] for client in client_info], axis=0)
                                # global_aggregator.create_dev_dataset({"dataset": dev_dataset, "label": dev_label})
                                # global_aggregator.create_dev_dataset({"dataset": dev_dataset})
                                
                                # Choose clients to train
                                # random.seed(round*1234)
                                # num_participants = random.uniform(0,1)
                                
                                selected_idx = random.sample([i for i in range(len(client_info))], int(num_participants*len(client_info)))  # 10 থেকে 5 client select করে
                                selected_clients = [client_info[i] for i in selected_idx]  # selected client list
                                
                                total_training_samples = sum([len(client['train_loader'].dataset) for client in selected_clients])  # মোট training sample count
                                
                                client_weights = []  # local weights list
                                # if round == 0:
                                for i, client in enumerate(selected_clients):  # selected client train loop
                                    logging.info("Training local model...")  # local training message
                                    # ClientTrainer বানায়
                                    device_trainer = ClientTrainer(model=global_aggregator.model, \
                                        save_dir=client['save_dir'], epoch=epoch, update_type=update_type, lr_rate=lr_rate)  # trainer settings দেয়
                                    device_trainer.run(client["train_loader"], client["valid_loader"])  # local model train করে
                                    # client_weights.append(copy.deepcopy(device_trainer.model.state_dict()))
                                    client_weights.append((copy.deepcopy(device_trainer.model.state_dict()), total_training_samples, len(client["train_loader"].dataset)))  # trained weights collect করে
                                    logging.info(f"Client {i} training done!")  # client training done message
                                
                                logging.info(f"Round {round+1}/{num_rounds} - Updating global model")  # global update শুরু
                                
                                # client_weights = random.sample(client_weights, int(num_participants * len(client_weights)))
                                global_aggregator.update(local_models=client_weights)  # MSEAvg দিয়ে global model বানায়

                                # global model update message
                                logging.info(f"Round {round+1}/{num_rounds} - Updated global model - \
                                    Global loss: {global_aggregator.val_loss}")  # global loss দেখায়
                                
                                logging.info("Training done! Evaluating...")  # evaluation শুরু
                                # evaluate the model in clients
                            
                                evaluator = Evaluator(global_aggregator.model, metric=metric, model_type=model_type)  # Evaluator বানায়
                                round_results = {}  # round result রাখে
                                for i, client in enumerate(client_info):  # সব client evaluate করে
                                    logging.info(f"Evaluating client {i} - name: {client['device']}")  # client evaluation message
                                    auc_score = evaluator.evaluate(client["test_loader"], client["train_loader"])  # value/data রাখে
                                    round_results[client['device']] = auc_score  # client AUC save করে
                                round_results["global_loss"] = global_aggregator.val_loss  # global loss save করে
                                round_results['join_clients'] = selected_idx  # joined clients save করে
                                round_results = {f'round_{round+1}': round_results}  # round result গুছায়
                                
                                # Append to the JSON file
                                with open(filename, 'a') as f:  # result file খোলে
                                    f.write(json.dumps(round_results) + '\n')  # result JSON-এ লেখে
                                
                                if global_aggregator.val_loss < min_val_loss:  # loss কমেছে কিনা দেখে
                                    min_val_loss = global_aggregator.val_loss  # best loss update করে
                                    global_worse = 0  # counter reset
                                
                                if global_aggregator.val_loss >= min_val_loss:  # loss improve হয়নি কিনা দেখে
                                    global_worse += 1  # counter বাড়ায়
                                    if global_worse > global_patience:  # patience শেষ কিনা দেখে
                                        logging.info("Early stopping in global round!")  # early stop message
                                        break  # training loop বন্ধ করে
                                    