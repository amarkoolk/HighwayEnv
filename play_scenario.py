import gymnasium as gym
import pygame
import numpy as np
import highway_env

def play():
    # Create the environment
    # render_mode="human" to open a window
    env = gym.make("highway-v0", render_mode="human")
    
    # Configure for manual control
    env.unwrapped.configure({
        "duration": 600,  # Longer duration
        "action": {
            "type": "ContinuousAction"
        },
        "simulation_frequency": 15,
        "policy_frequency": 15, # Same as sim freq for direct control
        "screen_width": 800,
        "screen_height": 400,
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False,
        "collision_reward": -1,
        "vehicles_count": 10, # Add some traffic
        "lanes_count": 4
    })
    
    print("Initializing environment...")
    obs, info = env.reset()
    done = False
    truncated = False
    
    print("\n" + "="*40)
    print("CONTROLS:")
    print("UP/DOWN: Accelerate/Decelerate")
    print("LEFT/RIGHT: Steer")
    print("ESC: Quit")
    print("="*40 + "\n")
    
    clock = pygame.time.Clock()
    
    while not (done or truncated):
        # Handle events
        keys = pygame.key.get_pressed()
        
        acc = 0.0
        steer = 0.0
        
        if keys[pygame.K_UP]:
            acc = 0.5
        elif keys[pygame.K_DOWN]:
            acc = -0.5
            
        if keys[pygame.K_LEFT]:
            steer = -0.5 # Left
        elif keys[pygame.K_RIGHT]:
            steer = 0.5 # Right
            
        action = np.array([acc, steer])
        
        # Step
        obs, reward, done, truncated, info = env.step(action)
        
        # Render
        env.render()
        
        # Check for collision classification
        vehicle = env.unwrapped.vehicle
        if vehicle.crashed:
            if hasattr(vehicle, 'collision_classification') and vehicle.collision_classification:
                print(f"\nCRASH DETECTED!")
                print(f"Type: {vehicle.collision_classification.collision_type}")
                print(f"Contact: {vehicle.collision_classification.contact_type}")
                print(f"Ego Feature: {vehicle.collision_classification.ego_feature}")
                print(f"NPC Feature: {vehicle.collision_classification.npc_feature}")
                
                # Optional: Pause on crash to see output
                # import time
                # time.sleep(1)
                
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                done = True
                
        clock.tick(15) # Limit FPS
                
    env.close()
    print("Simulation finished.")

if __name__ == "__main__":
    play()
