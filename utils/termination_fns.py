import numpy as np

def obs_unnormalization(termination_fn, obs_mean, obs_std):
    def thunk(obs, act, next_obs):
        obs = obs*obs_std + obs_mean
        next_obs = next_obs*obs_std + obs_mean
        return termination_fn(obs, act, next_obs)
    return thunk

def termination_fn_halfcheetah(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    not_done = np.logical_and(np.all(next_obs > -100, axis=-1), np.all(next_obs < 100, axis=-1))
    done = ~not_done
    done = done[:, None]
    return done

def termination_fn_hopper(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done =  np.isfinite(next_obs).all(axis=-1) \
                    * np.abs(next_obs[:,1:] < 100).all(axis=-1) \
                    * (height > .7) \
                    * (np.abs(angle) < .2)

    # print(height, angle)

    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_halfcheetahveljump(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False]).repeat(len(obs))
    done = done[:,None]
    return done

def termination_fn_antangle(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    x = next_obs[:, 0]
    not_done = 	np.isfinite(next_obs).all(axis=-1) \
                * (x >= 0.2) \
                * (x <= 1.0)

    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_ant(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    x = next_obs[:, 0]
    not_done = 	np.isfinite(next_obs).all(axis=-1) \
                * (x >= 0.2) \
                * (x <= 1.0)

    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_walker2d(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done =  np.logical_and(np.all(next_obs > -100, axis=-1), np.all(next_obs < 100, axis=-1)) \
                * (height > 0.8) \
                * (height < 2.0) \
                * (angle > -1.0) \
                * (angle < 1.0)
    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_point2denv(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False]).repeat(len(obs))
    done = done[:,None]
    return done

def termination_fn_point2dwallenv(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False]).repeat(len(obs))
    done = done[:,None]
    return done

def termination_fn_pendulum(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.zeros((len(obs), 1))
    return done

def termination_fn_humanoid(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    z = next_obs[:,0]
    done = (z < 1.0) + (z > 2.0)

    done = done[:,None]
    return done

def termination_fn_pen(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    obj_pos = next_obs[:, 24:27]
    done = obj_pos[:, 2] < 0.075

    done = done[:,None]
    return done

def terminaltion_fn_door(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False] * obs.shape[0])

    done = done[:, None]
    return done

def termination_fn_go1(obs, act, next_obs):
    dof_limit = np.array([[33.5, -0.802851455917, 0.802851455917, 50],
                        [33.5, -1.0471975512, 4.18879020479, 28],
                        [33.5, -2.69653369433, -0.916297857297, 28],
                        [33.5, -0.802851455917, 0.802851455917, 50],
                        [33.5, -1.0471975512, 4.18879020479, 28],
                        [33.5, -2.69653369433, -0.916297857297, 28],
                        [33.5, -0.802851455917, 0.802851455917, 50],
                        [33.5, -1.0471975512, 4.18879020479, 28],
                        [33.5, -2.69653369433, -0.916297857297, 28],
                        [33.5, -0.802851455917, 0.802851455917, 50],
                        [33.5, -1.0471975512, 4.18879020479, 28],
                        [33.5, -2.69653369433, -0.916297857297, 28]]).astype(np.float32)
    """ self.dof, 
        self.dof_vel,
        self.action,"""
    print(dof_limit[:,2] )
    out1 = next_obs[:,0:12] >  dof_limit[:,2]  
    out2 = next_obs[:,0:12] < dof_limit[:,1] 
    out3 = next_obs[:,12:24] > dof_limit[:,0]  
    out4 = next_obs[:,12:24] < -dof_limit[:,0]

    result = out1 | out2  
    result = result | out3
    result = result | out4
    
    return np.any(result, axis=1).reshape(-1, 1)
    #return next_obs[:][0:12]>dof_limit[:][2] | next_obs[:][0:12]<dof_limit[:][1] | next_obs[:][12:24]>dof_limit[:][0] | next_obs[:][12:24]<-dof_limit[:][0]

def termination_fn_aliengo(obs, act, next_obs):
    # [[FR_hip, FR_Thigh, FR_Calf],
    #  [FL_hip, FL_Thigh, FL_Calf],
    #  [RR_hip, RR_Thigh, RR_Calf],
    #  [RL_hip, RL_Thigh, RL_Calf]]
    dof_limit = np.array([[33.5, -0.873, 1.047, 20],
                        [33.5, -0.524, 3.927, 20],
                        [33.5, -2.775, -0.611, 20],
                        [33.5, -0.873, 1.047, 20],
                        [33.5, -0.524, 3.927, 20],
                        [33.5, -2.775, -0.611, 20],
                        [33.5, -0.873, 1.047, 20],
                        [33.5, -0.524, 3.927, 20],
                        [33.5, -2.775, -0.611, 20],
                        [33.5, -0.873, 1.047, 20],
                        [33.5, -0.524, 3.927, 20],
                        [33.5, -2.775, -0.611, 20]]).astype(np.float32)
    
    dof_pos = next_obs[:, 18:30] # indices for dof_pos state
    dof_vel = next_obs[:, 30:42] # indices for dof_vel state

    # Check limit violation
    # Return TRUE in case of violation
    out1 = dof_pos  >  dof_limit[:, 2]
    out2 = dof_pos  <  dof_limit[:, 1]
    out3 = dof_vel  >  dof_limit[:, 0]
    out4 = dof_vel  < -dof_limit[:, 0]

    result = out1 | out2 | out3 | out4
    return result

def termination_fn_aliengo_v1(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    # body_height = next_obs[:, 6]      # commanded body height
    # body_pitch  = next_obs[:, 13]     # commanded pitch
    # body_roll   = next_obs[:, 14]     # commanded roll

    # not_done =  np.isfinite(next_obs).all(axis=-1) \
    #                 * np.abs(next_obs[:,1:] < 100).all(axis=-1) \
    #                 * (height > .7) \
    #                 * (np.abs(angle) < .2)

    # print(height, angle)

    # Orientation check via gravity vector (robot flipped if z-component is too small/negative)
    gravity_z = next_obs[:, 2]                 # z-component of gravity vector
    orientation_violation = gravity_z > -0.6   # fell if tilted too far

    done = orientation_violation[:, None]      # shape (batch, 1) for consistency
    return done

def termination_fn_aliengo_v2(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    # Extract DOF positions and velocities
    dof_pos = next_obs[:, 18:30]
    dof_vel = next_obs[:, 30:42]

    # Aliengo joint limits
    pos_upper = np.array([ 1.047,  3.927, -0.611] * 4)   # rad
    pos_lower = np.array([-0.873, -0.524, -2.775] * 4)   # rad
    vel_limit = 20                                       # rad/s
    torque_limit = 33.5                                  # Nm

    # Check joint violations
    joint_pos_violation = np.any((dof_pos > pos_upper) | (dof_pos < pos_lower), axis=1)
    joint_vel_violation = np.any(np.abs(dof_vel) > vel_limit, axis=1)

    # Orientation check via gravity vector (robot flipped if z-component is too small/negative)
    gravity_z = next_obs[:, 2]
    orientation_violation = abs(gravity_z) < 0.6   # fell if tilted too far

    # Combine all conditions
    done = joint_pos_violation | joint_vel_violation | orientation_violation
    return done[:, None]

def get_termination_fn(task):
    if 'halfcheetahvel' in task:
        return termination_fn_halfcheetahveljump
    elif 'halfcheetah' in task:
        return termination_fn_halfcheetah
    elif 'hopper' in task:
        return termination_fn_hopper
    elif 'antangle' in task:
        return termination_fn_antangle
    elif 'ant' in task:
        return termination_fn_ant
    elif 'walker2d' in task:
        return termination_fn_walker2d
    elif 'point2denv' in task:
        return termination_fn_point2denv
    elif 'point2dwallenv' in task:
        return termination_fn_point2dwallenv
    elif 'pendulum' in task:
        return termination_fn_pendulum
    elif 'humanoid' in task:
        return termination_fn_humanoid
    elif 'pen' in task:
        return termination_fn_pen
    elif 'door' in task:
        return terminaltion_fn_door
    elif 'go1' in task:
        #return termination_fn_go1
        return termination_fn_hopper
    elif 'aliengo' in task:
        #return termination_fn_aliengo
        return termination_fn_aliengo_v1
    else:
        raise np.zeros
