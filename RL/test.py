

def test_model(model, env, test_times):
    obs = env.reset()
    done = False
    R = 0
    while not done:
        action = model.return_action(obs)
        obs, Reward, done, _ = env.step(action) 
        R += Reward

    return R
