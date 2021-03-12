import wandb

wandb.init(project="action-embedding-transformer")


for epoch in range(10):
    loss = 0 # change as appropriate :)
    wandb.log({'epoch': epoch, 'loss': loss})

wandb.log({"video": wandb.Video('teste.gif', fps=30, format="gif")})
