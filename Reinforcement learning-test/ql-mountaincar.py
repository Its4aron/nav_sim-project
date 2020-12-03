import gym
import numpy as np

LEARNING_RATE = 0.1
DISCOUNT = 0.95 #weight (future reward vs current reward)
EPISODES = 25000 #Episodes to learn 
SHOW_EVERY = 100 #render every SHOW_EVERY-th episode
env = gym.make("MountainCar-v0") #Enviorment for testing

DISCRETE_OS_SIZE = [20] *len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

#EPSILON
epsilon = 0.5 #Higher the epsilon the more exploration
S_DECAYING_EPSILON = 1
E_DECAYING_EPSILON = EPISODES // 2 
decayed_epsilon = epsilon/(E_DECAYING_EPSILON - S_DECAYING_EPSILON)

print(discrete_os_win_size) #range
q_table = np.random.uniform(low=-2,high=0,size=(DISCRETE_OS_SIZE + [env.action_space.n])) #20 by 20 table that will contain every combination of Positon + Velocity 
print(q_table.shape) #3d table

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int)) #tuple int 



for episode in range(EPISODES):
    if not (episode % SHOW_EVERY):
        render = True
        print(episode,"<- current episode")
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:

        #if np.random.random() > epsilon:
        action = np.argmax(q_table[discrete_state]) #get the max action for current combo
        #else:
          #  action = np.random.randint(0,env.action_space.n)

        new_state,reward,done,_ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        
        if render:
            env.render()
        if not done:
             max_future_q = np.max(q_table[new_discrete_state])
             current_q = q_table[discrete_state + (action,)]
             new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q) #forumla to calc the future q
             q_table[discrete_state+(action,)] = new_q # update our current q with our new q

        elif new_state[0] >= env.goal_position:
            print("Reached goal on episode {}".format(episode))
            q_table[discrete_state + (action,)] = 0 #reward with max q

        discrete_state = new_discrete_state
    if E_DECAYING_EPSILON >= episode >= S_DECAYING_EPSILON:
        epsilon -= decayed_epsilon

        
env.close()