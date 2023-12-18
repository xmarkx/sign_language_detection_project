import click






@click.command()
@click.option('-c', '--collect', is_flag=True, help='Use the --collect flag to run the data collection')

def main(collect):
    if collect:
        # actions_to_collect = input('The actions to collect: ')
        # # Your logic for data collection with actions
        # l = actions_to_collect.split()  # Convert space-separated actions to a list
        # print(l)
        if collector_instance in locals():
            print('I exist')
        else:
            print('I dont exist')

if __name__ == '__main__':
    main()

