from training import MLE
import pickle

lamb = [0.5, 0.1, 0.05]
learn = [0.1, 0.05, 0.01]

val_loss_complete = []
val_loss = []
val_t = []
val_tstar = []

# grid search for initial learning rate and lambda hyperparameter
# checks the number of timesteps until convergence and the validation loss it converges too
for i, l in enumerate(lamb):
    val_loss.append([])
    val_loss.append([])
    val_tstar.append([])
    val_loss_complete.append([])
    for j, y in enumerate(learn):
        weight, avg_loss, val_loss, t, tstar = MLE.stochastic_gradient_descent(25, y, 10.0, 1, 4, 25, True, l)
        val_loss[i].append(val_loss[-1])
        val_t[i].append(t)
        val_tstar[i].append(tstar)
        val_loss_complete[i].append(val_loss)

pickle.dump(val_loss, open("grid_search_loss.gr", "wb"))
pickle.dump(val_loss_complete, open("grid_search_loss_complete.gr", "wb"))
pickle.dump(val_t, open("grid_search_t.gr", "wb"))
pickle.dump(val_tstar, open("grid_search_tstar.gr", "wb"))

print(val_loss)
print(val_t)
print(val_tstar)





