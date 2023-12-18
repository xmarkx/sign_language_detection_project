import collector
import model2
import inference2
import click
from tensorflow.keras.models import load_model
import cv2


@click.command()
@click.option('-c', '--collect', is_flag=True, help='Use the --collect flag to run the data collection')
@click.option('-t', '--train', is_flag=True, help='Use the --train flag to run the training')
@click.option('-i', '--inference', is_flag=True, help='Use the --inference flag to run inference')


def main(collect, train, inference):
    if collect:
        print('Starting action collection...')
        actions_to_collect = input('Provide the actions to collect (use space inbetween if multiple actions are provided): ')
        collector_instance = collector.Collector()        
        collector_instance.actions = actions_to_collect.split()
        print(f"You provided the following actions to collect: {collector_instance.actions}, it's length is {len(collector_instance.actions)}")
        print(collector_instance.no_sequences, collector_instance.sequence_length)
        collector_instance.folder_setup(actions_in=collector_instance.actions, no_sequences_in=collector_instance.no_sequences)
        collector_instance.collect_data()

    if train:
        data_choice = input('Which data do You want to train on (c / d)? (c → data from the current data collection / d → default data: english alphabet characters): ').lower()
        assert data_choice in ['c', 'd'], "Please provide c or d characters for using 'c'urrently collected data or 'd'efault collected data"
        if data_choice == 'c':
            model_name = input("Please give a name to Your model: ").lower()
            model_name = model_name + ".keras"
            model_instance = model2.Model(data=collector_instance, model_name=model_name)
        elif data_choice == 'd':
            model_instance = model2.Model()
        print('Starting training...')
        model_instance.train()
        model_instance.evaluate()

    if inference:
        if train and data_choice == 'c':
            model_name = model_instance.model_name
        else:
            model_choice = input("Which model do You want to run inference on ('d'efault model / custom model_name): ").lower()
            if model_choice == 'd':
                model = model2.Model()
                model_name = model.model_name
                print(f'Using default model from Model.model_name: {model_name}')
            else:
                model_name = model_choice

        # here we have the model_name already
        try:
            inference2.run_inference(model_name=model_name, collector=collector_instance)
        except Exception as e:
            print(e)
            print('Using the default Collector instance')
            inference2.run_inference(model_name=model_name, collector=collector.Collector())

            
if __name__ == '__main__':
    main()
