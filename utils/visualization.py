import matplotlib.pyplot as plt

def draw_loss(loss_list: list, save_filename: str):
    fig = plt.figure(figsize=(8, 6))
    plt.title("Loss", fontsize='16')
    plt.plot(list(range(len(loss_list))), loss_list)
    plt.xlabel("epoch", fontsize='13')
    plt.ylabel("loss", fontsize='13')
    plt.savefig(save_filename)