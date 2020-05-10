import matplotlib.pyplot as plt
import re

class Drawer:
    def __init__(self, x_dim = 500):
        self.models_train_loss = {}
        self.models_valid_loss = {}
        self.iterations = [i for i in range(x_dim)]

    def add_model_log(self, filename, modelname):
        train_loss = []
        valid_loss = []
        with open(filename) as fp:
            Lines = fp.readlines()
            for line in Lines:
                tokens = list(filter(None, re.split(', | |: |\n', line)))
                if len(tokens) == 13 and tokens[5] == 'iterations':
                    train_loss.append(float(tokens[9]))
                    valid_loss.append(float(tokens[12]))

        self.models_train_loss[modelname] = train_loss
        self.models_valid_loss[modelname] = valid_loss

    def compare_and_draw(self, models, mode = 'valid'):
        if mode == 'valid':
            for model in models:
                plt.plot(self.iterations, self.models_valid_loss[model], label = model)
            plt.legend()
            plt.show()

        elif mode == 'train':
            for model in models:
                plt.plot(self.iterations, self.models_train_loss[model], label=model)
            plt.legend()
            plt.show()

        elif mode == 'all':
            for model in models:
                plt.plot(self.iterations, self.models_valid_loss[model], label=model+"_valid")
                plt.plot(self.iterations, self.models_train_loss[model], label=model+"_train")
            plt.legend()
            plt.show()


if __name__ == '__main__':
    # default number of iterations is 500
    drawer = Drawer(x_dim=500)
    drawer.add_model_log(filename="log_beauty_crosattn_cff_bert.txt", modelname="crossattn_cff")
    drawer.compare_and_draw(models=["crossattn_cff"], mode="all")


