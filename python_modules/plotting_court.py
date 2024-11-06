import matplotlib.pyplot as plt
import matplotlib.patches as patches

#THIS CODE WAS FULLY GENERATED USING CHATGPT. I CLAIM NO OWNERSHIP OF THIS CODE. THE CODE IS USED TO PLOT THE BASKETBALL COURT 

def plot_court(ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    # Court outline
    court = patches.Rectangle((0, 0), width=94, height=50, color='black', fill=False, linewidth=2)
    ax.add_patch(court)
    
    # Center circle
    center_circle = plt.Circle((47, 25), 6, color='black', fill=False, linewidth=2)
    ax.add_patch(center_circle)
    
    # Left half-court
    left_paint = patches.Rectangle((0, 17), width=19, height=16, color='black', fill=False, linewidth=2)
    ax.add_patch(left_paint)
    
    left_free_throw_circle = plt.Circle((19, 25), 6, color='black', fill=False, linestyle='dashed', linewidth=2)
    ax.add_patch(left_free_throw_circle)
    
    left_hoop = plt.Circle((5.25, 25), 0.75, color='black', fill=False, linewidth=2)
    ax.add_patch(left_hoop)
    
    left_backboard = plt.Line2D([4, 4], [22, 28], color='black', linewidth=2)
    ax.add_line(left_backboard)
    
    left_three_point_arc = patches.Arc((5.25, 25), 47.5, 47.5, theta1=-68, theta2=68, color='black', linewidth=2)
    ax.add_patch(left_three_point_arc)
    
    # Right half-court
    right_paint = patches.Rectangle((75, 17), width=19, height=16, color='black', fill=False, linewidth=2)
    ax.add_patch(right_paint)
    
    right_free_throw_circle = plt.Circle((75, 25), 6, color='black', fill=False, linestyle='dashed', linewidth=2)
    ax.add_patch(right_free_throw_circle)
    
    right_hoop = plt.Circle((88.75, 25), 0.75, color='black', fill=False, linewidth=2)
    ax.add_patch(right_hoop)
    
    right_backboard = plt.Line2D([90, 90], [22, 28], color='black', linewidth=2)
    ax.add_line(right_backboard)
    
    right_three_point_arc = patches.Arc((88.75, 25), 47.5, 47.5, theta1=112, theta2=248, color='black', linewidth=2)
    ax.add_patch(right_three_point_arc)
    
    # Half-court line
    half_court_line = plt.Line2D([47, 47], [0, 50], color='black', linewidth=2)
    ax.add_line(half_court_line)
    
    # Set limits and remove axes
    ax.set_xlim(0, 94)
    ax.set_ylim(0, 50)
    ax.axis('off')
    plt.gca().set_aspect('equal', adjustable='box')
