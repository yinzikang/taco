#!/bin/bash
#python train_fpv_asymmetry_ppo.py --train_mode=train --task_mode=pos --lenObservations=1 --lenStates=5 --use_actor_encoder=False --use_critic_encoder=True --critic_encoder_type=LSTM --rotor_response_time=0.017 --delay_time=40 --lipschitz_para=4&  # best
#wait
#python train_fpv_asymmetry_ppo.py --train_mode=train --task_mode=rotate --lenObservations=1 --lenStates=5 --use_actor_encoder=False --use_critic_encoder=True --critic_encoder_type=LSTM --rotor_response_time=0.017 --delay_time=40 --lipschitz_para=4&
#wait
#python train_fpv_asymmetry_ppo.py --train_mode=train --task_mode=flip --lenObservations=1 --lenStates=5 --use_actor_encoder=False --use_critic_encoder=True --critic_encoder_type=LSTM --rotor_response_time=0.017 --delay_time=40 --lipschitz_para=4&
#wait
#python train_fpv_asymmetry_ppo.py --train_mode=train --task_mode=roll --lenObservations=1 --lenStates=5 --use_actor_encoder=False --use_critic_encoder=True --critic_encoder_type=LSTM --rotor_response_time=0.017 --delay_time=40 --lipschitz_para=4&
#wait
python train_fpv_asymmetry_ppo.py --train_mode=train --task_mode=mix --lenObservations=1 --lenStates=5 --use_actor_encoder=False --use_critic_encoder=True --critic_encoder_type=LSTM --rotor_response_time=0.017 --delay_time=40 --lipschitz_para=4&
wait
