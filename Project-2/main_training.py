from training import MLE
import pickle

subset = 1  # 1 tom, 2 nedko, 3 nuno

lamb = [0.5, 0.1, 0.05]
if subset == 1:
    lamb = [1.0, 0.5]
    learn = [1.0, 0.5]
if subset == 2:
    learn = [0.1]
if subset == 3:
    learn = [0.01]

val_loss_complete = []
val_loss = []
val_t = []
val_tstar = []
avg_loss_complete = []

# grid search for initial learning rate and lambda hyperparameter
# checks the number of timesteps until convergence and the validation loss it converges too
for i, l in enumerate(lamb):
    val_loss.append([])
    val_t.append([])
    val_tstar.append([])
    val_loss_complete.append([])
    avg_loss_complete.append([])

    for j, y in enumerate(learn):
        if i == 1 and j == 1:
            break
        # input: batch-size, initial learning rate, convergence threshold, number of convergence checks,
        # number of batches, max epochs, whether to regularize, lambda
        weight, avg_loss, val, t, tstar = MLE.stochastic_gradient_descent(25, y, 10.0, 1, 4, 25, True, l)
        val_loss[i].append(val[-1])
        val_t[i].append(t)
        val_tstar[i].append(tstar)
        val_loss_complete[i].append(val_loss)
        avg_loss_complete[i].append(avg_loss)

pickle.dump(val_loss, open("grid_search_loss_" + str(learn[0]) + ".gr", "wb"))
pickle.dump(val_loss_complete, open("grid_search_loss_complete_" + str(learn[0]) + ".gr", "wb"))
pickle.dump(avg_loss_complete, open("grid_search_avg_complete_" + str(learn[0]) + ".gr", "wb"))
pickle.dump(val_t, open("grid_search_t_" + str(learn[0]) + ".gr", "wb"))
pickle.dump(val_tstar, open("grid_search_tstar_" + str(learn[0]) + ".gr", "wb"))

print(val_loss)
print(val_t)
print(val_tstar)





