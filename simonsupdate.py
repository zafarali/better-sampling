# importance sampling

def draw_from_proposal(w, time_left, push_toward=0):
#     print('pos:',w)
#     print('toward:',push_toward)
    b = 1
    probs = np.array([np.exp(b*(push_toward-w)), np.exp(b*(w-push_toward)), np.exp(0.5*b)])
    probs /= np.sum(probs)
    choices = [1/2, -1/2, 0]
    choice = np.random.choice(choices, p=probs)
    prob = probs[choices.index(choice)]
    return choice, prob


def draw_from_proposal_minimal(w, time_left, push_toward=0):
#     print('pos:',w)
#     print('toward:',push_toward)
    b = 1
    # We want to push slightly, such that the average step is d/T, where d is distance to 
    # the acceptable position, and T is the number of generations left. 
    bias = 0 
    sign = w/np.abs(w)
    if np.abs(w) > np.abs(push_toward):
        bias = (sign*push_toward - w)*1./time_left 
    
    
    # with probability p, pick uniform sampling; with probability 1-p, pick a step in the bias direction. 
    # expected bias is (1-p) * step_size  
    step_size = .5
    if np.abs(bias)>step_size:
        #print("will fail, might as well stop now.")
        return 0,1
    p = 1-np.abs(bias)/step_size
    bias_prob_term = np.array([0,0,0])
    if bias > 0:
        bias_prob_term[2] = 1
    if bias < 0:
        bias_prob_term[0] = 1
    choices = [-step_size,0,step_size]
    probs = np.array([p/3.,p/3.,p/3.])+ (1-p)*bias_prob_term
    choice = np.random.choice(choices, p=probs)
    prob = probs[choices.index(choice)]
    return choice, prob



trajectories = []    

mc_samples = 500
c = 2
W_T_true = Ws[-1] # the position we observed.
T = len(Ws)
successful_trees = 0
endpoints = []
w0s = []
ones = []
for i in range(mc_samples):

    w_t = Ws[-1] # start at the end
    traj = [w_t]
    log_path_prob = 0 # use 1 because we can multiply easily
    log_q_prob = 0
    # go in reverse time:
    for t in reversed(range(0, T)):
        
        # draw a reverse step from the proposal
        step, prob = draw_from_proposal_minimal(w_t, t, push_toward=c)
        log_q_prob += np.log(prob)
        # this is p(w_{t} | w_{t+1})
        log_path_prob += p_prob_transition(step)
        
        # take the reverse step:
        w_t = w_t + step
        traj.append(w_t)
    log_path_prob += np.log(uniform.pdf(w_t, -c, c+c))
    trajectories.append(traj)
    if log_path_prob > -10**10:
        successful_trees += 1
        endpoints.append(w_t)
        w0s.append(w_t * np.exp(log_path_prob - log_q_prob))
        ones.append(1 * np.exp(log_path_prob - log_q_prob))

successful_trees /= mc_samples

print('Estimated w0:', np.mean(np.array(w0s)/np.mean(ones)), '| variance:',np.var(np.array(w0s)/np.mean(ones)))
print("prop, succesful trees", successful_trees)