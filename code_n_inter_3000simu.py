import numpy as np
import minidem as dem
import random
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd

# Fonction pour sauvegarder l'image de la simulation
def save_simulation_image(grains, filename, use_grain_color=False):
    fig, ax = plt.subplots()
    for gr in grains:
        color = 'white'
        if use_grain_color and hasattr(gr, 'color'):
            color = gr.color

        circle = plt.Circle((gr.pos[0], gr.pos[1]), gr.radius, color=color, ec='black')
        ax.add_artist(circle)

    if contact_list:
        max_force = max(np.linalg.norm(c.force) for c in contact_list)

        for c in contact_list:
            fmag = np.linalg.norm(c.force)
            if fmag == 0:
                continue
            thickness = 3 + 2 * (fmag / max_force)

            x1, y1 = c.grain1.pos[0], c.grain1.pos[1]

            # Cas paroi-grain
            if getattr(c, "wall_pos", None) in ["left", "right", "top"]:
                wp = c.wall_pos

                if wp == "left":
                    x2, y2 = 0, y1
                elif wp == "right":
                    x2, y2 = 100, y1
                elif wp == "top":
                    x2, y2 = x1, 100
                else:
                    x2, y2 = c.grain2.pos[0], c.grain2.pos[1]
            else:
                x2, y2 = c.grain2.pos[0], c.grain2.pos[1]

            ax.plot([x1, x2], [y1, y2], 'r-', lw=thickness)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal', 'box')
    plt.axis('on')
    outpath = os.path.abspath(filename)
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Image saved to:", outpath)


# Fonction principale pour exécuter la simulation
def run_simulation(simulation_id):
    global gray_grain_list, gray_grain_list_down, gray_grain_list_top, gray_grain_list_left, gray_grain_list_right, contact_list
    global CURRENT_SIM_ID
    CURRENT_SIM_ID = simulation_id

    # Arrêter l'animation précédente si elle existe
    if 'dem.animate' in globals():
        dem.animate.event_source.stop()

    # Réinitialiser la figure et les axes
    dem.simu.fig, dem.simu.ax = plt.subplots()
    dem.simu.ax.set_xlim(0, 100)
    dem.simu.ax.set_ylim(0, 100)
    dem.simu.ax.set_aspect('equal', adjustable='box')

    # Réinitialiser l'environnement de simulation
    plt.close('all')
    dem.simu.grain_list.clear()
    dem.simu.bond_list.clear()
    dem.simu.patch_list.clear()
    dem.simu._init_plot = False
    dem.simu.current_iter_number = 0
    dem.simu.t = 0.

    # Réinitialisation des listes pour la nouvelle simulation
    gray_grain_list = []
    gray_grain_list_down = []
    gray_grain_list_top = []
    gray_grain_list_left = []
    gray_grain_list_right = []
    contact_list = []

    # Configuration initiale des grains
    rad = 5
    density = 1
    for x in range(rad, 100, 2 * rad):
        for y in range(2 * rad, 100 - rad, 4 * rad):
            gr = None
            if y == 2 * rad:  # first layer
                gr = dem.grain(dem.vec(x, y), rad, density)
                gr.color = "gray"
                gray_grain_list_down.append(gr)
                gray_grain_list.append(gr)

            if gr is None:
                x_rand = x + random.random() - 0.5
                y_rand = y + random.random() - 0.5
                rad_rand = (rad + 3) - random.randrange(1, 6, 2)
                gr = dem.grain(dem.vec(x_rand, y_rand), rad_rand, density)
                gr.vel = dem.vec((random.random() - 0.5) * 10., (random.random() - 0.5) * 10.)

    # Définition de la durée et des autres paramètres de la simulation
    dem.simu.dt = 0.001
    output_video_name = os.path.join("output", f"simulation_{simulation_id}.mp4")

    # Exécution de la simulation
    equilibrium_reached = dem.run(
        tot_iter_number=5000,
        update_plot_each=10,
        loop_fn=time_loop,
        video_name=output_video_name
    )

    if equilibrium_reached:
        print(f"Simulation {simulation_id} terminée avec équilibre atteint.")
    else:
        print(f"Simulation {simulation_id} terminée sans atteindre l'équilibre.")

    print(f"Fin de la simulation {simulation_id}, le temps écoulé est {dem.simu.t} s")


def extrapoint():
    global biggrain
    if dem.simu.current_iter_number == 1:
        intersection = [x for x in dem.simu.grain_list if x not in gray_grain_list_down]
        biggrain = random.choice(intersection)
        biggrain.radius = 15


def add_extern_force():
    tot_force = 20e4

    # manage down wall
    tot_force_down = tot_force
    for gr in gray_grain_list_down:
        tot_force_down += gr.force[1]
    for gr in gray_grain_list_down:
        gr.force[1] = tot_force_down / len(gray_grain_list_down)


def apply_boundary_condition():
    for gr in gray_grain_list_down:
        gr.pos[0] = gr.initial_pos[0]
        gr.vel[0] = 0.


def add_gravity_force():
    for gr in dem.simu.grain_list:
        gr.force += gr.mass * dem.vec(0., -9.81)


gamma = 1000

def add_dissipation():
    for gr in dem.simu.grain_list:
        gr.force += -gamma * gr.vel


def rigid_wall():
    f = 1.
    for gr in dem.simu.grain_list:
        if gr.pos[0] - gr.radius < 0:
            gr.pos[0] = gr.radius
            if gr.vel[0] < 0.:
                gr.vel[0] *= -f
        elif gr.pos[0] + gr.radius > 100:
            gr.pos[0] = 100 - gr.radius
            if gr.vel[0] > 0.:
                gr.vel[0] *= -f

        if gr.pos[1] - gr.radius < 0:
            gr.pos[1] = gr.radius
            if gr.vel[1] < 0.:
                gr.vel[1] *= -f
        elif gr.pos[1] + gr.radius > 100:
            gr.pos[1] = 100 - gr.radius
            if gr.vel[1] > 0.:
                gr.vel[1] *= -f


def reset_force():
    for gr in dem.simu.grain_list:
        gr.force = dem.vec(0., 0.)


def velocity_verlet():
    dt = dem.simu.dt
    for gr in dem.simu.grain_list:
        a = gr.force / gr.mass
        gr.vel += (gr.acc + a) * (dt / 2.)
        gr.pos += gr.vel * dt + 0.5 * a * (dt ** 2.)
        gr.acc = a


class ContactInfo:
    def __init__(self, grain1, grain2, force, wall_pos=None):
        self.grain1 = grain1
        self.grain2 = grain2
        self.force = force
        self.wall_pos = wall_pos


class Wall:
    def __init__(self, pos):
        self.pos = np.array(pos)
        self.radius = 0
        self.wall = True


contact_list = []

youngs_modulus = 1e3
plastic_deformation_limit = 0.01

def calculate_contact_force(gr1, gr2):
    dist = np.linalg.norm(gr2.pos - gr1.pos)

    if dist < 1e-12:
        return np.array([0.0, 0.0])

    overlap = gr1.radius + gr2.radius - np.linalg.norm(gr2.pos - gr1.pos)
    if overlap > 0:
        normal_force_magnitude = youngs_modulus * overlap
        if overlap > plastic_deformation_limit:
            normal_force_magnitude *= (plastic_deformation_limit / overlap)

        normal_direction = (gr2.pos - gr1.pos) / np.linalg.norm(gr2.pos - gr1.pos)
        total_force = normal_direction * normal_force_magnitude
        return total_force
    else:
        return np.array([0.0, 0.0])


def add_mirror_grains_and_contacts():
    global contact_list

    for gr in dem.simu.grain_list:

        # Mur gauche
        if gr.pos[0] - gr.radius < 0:
            mirror_pos = np.array([-gr.pos[0], gr.pos[1]])
            dist = np.linalg.norm(gr.pos - mirror_pos)
            overlap = 2 * gr.radius - np.linalg.norm(gr.pos - mirror_pos)

            if overlap > 0:
                if dist < 1e-12:
                    normal_direction = np.array([1., 0.])
                else:
                    normal_direction = (gr.pos - mirror_pos) / np.linalg.norm(gr.pos - mirror_pos)

                force = youngs_modulus * overlap * normal_direction
                gr.force += force

                fake_grain = type("wall", (), {"pos": mirror_pos})
                contact_list.append(ContactInfo(gr, fake_grain, force, wall_pos="left"))

        # Mur droite
        if gr.pos[0] + gr.radius > 100:
            mirror_pos = np.array([200 - gr.pos[0], gr.pos[1]])
            dist = np.linalg.norm(gr.pos - mirror_pos)
            overlap = 2 * gr.radius - np.linalg.norm(gr.pos - mirror_pos)

            if overlap > 0:
                if dist < 1e-12:
                    normal_direction = np.array([-1., 0.])
                else:
                    normal_direction = (gr.pos - mirror_pos) / np.linalg.norm(gr.pos - mirror_pos)

                force = youngs_modulus * overlap * normal_direction
                gr.force += force

                fake_grain = type("wall", (), {"pos": mirror_pos})
                contact_list.append(ContactInfo(gr, fake_grain, force, wall_pos="right"))

        # Mur haut
        if gr.pos[1] + gr.radius > 100:
            mirror_pos = np.array([gr.pos[0], 200 - gr.pos[1]])
            dist = np.linalg.norm(gr.pos - mirror_pos)
            overlap = 2 * gr.radius - np.linalg.norm(gr.pos - mirror_pos)

            if overlap > 0:
                if dist < 1e-12:
                    normal_direction = np.array([0., -1.])
                else:
                    normal_direction = (gr.pos - mirror_pos) / np.linalg.norm(gr.pos - mirror_pos)

                force = youngs_modulus * overlap * normal_direction
                gr.force += force

                fake_grain = type("wall", (), {"pos": mirror_pos})
                contact_list.append(ContactInfo(gr, fake_grain, force, wall_pos="top"))


def manage_contact_and_update_forces():
    global contact_list
    contact_list = []
    l = dem.lcm.compute_colliding_pair()
    for (gr1, gr2) in l:
        if not (gr1 in gray_grain_list and gr2 in gray_grain_list):
            dem.contact(gr1, gr2)
            force = calculate_contact_force(gr1, gr2)
            contact_list.append(ContactInfo(gr1, gr2, force))
            gr1.force -= force
            gr2.force += force


def calculate_total_force():
    for gr in dem.simu.grain_list:
        gr.total_force = np.linalg.norm(gr.force)


def update_grain_colors():
    max_force_by_grain = {}
    for contact in contact_list:
        force_magnitude = np.linalg.norm(contact.force)
        for grain in [contact.grain1, contact.grain2]:
            if grain not in max_force_by_grain:
                max_force_by_grain[grain] = force_magnitude
            else:
                max_force_by_grain[grain] = max(max_force_by_grain[grain], force_magnitude)

    max_force_global = max(max_force_by_grain.values(), default=0)

    for grain, max_force in max_force_by_grain.items():
        normalized_force = max_force / max_force_global if max_force_global > 0 else 0
        gray_shade = 1 - normalized_force
        grain.color = [gray_shade, gray_shade, gray_shade]


# Créer le dossier s'il n'existe pas
os.makedirs("output", exist_ok=True)


def save_nodes(simulation_id):
    """Sauvegarde les positions et rayons dans un fichier CSV"""
    data = []
    for idx, gr in enumerate(dem.simu.grain_list, start=1):
        x, y = gr.pos
        R = gr.radius
        data.append([idx, x, y, R])

    df = pd.DataFrame(data, columns=["grain_id", "x", "y", "R"])

    filename = os.path.join("output", f"nodes_{simulation_id}.csv")
    df.to_csv(filename, index=False)

    print(f"Nodes sauvegardé : {filename}")


def save_forces(simulation_id):
    data = []

    for c in contact_list:
        fij = np.linalg.norm(c.force)
        if fij <= 0:
            continue

        i = dem.simu.grain_list.index(c.grain1) + 1
        is_wall_contact = getattr(c, "wall_pos", None) is not None

        if is_wall_contact:
            j = "a"
            contact_name = f"f{i}a"
            wall_pos = c.wall_pos if c.wall_pos else ""
        else:
            j = dem.simu.grain_list.index(c.grain2) + 1
            contact_name = f"f{i}{j}"
            wall_pos = ""

        data.append([contact_name, i, j, wall_pos, fij])

    df = pd.DataFrame(
        data,
        columns=["contact", "grain_i", "grain_j", "wall_pos", "fij"]
    )

    filename = os.path.join("output", f"forces_{simulation_id}.csv")
    df.to_csv(filename, index=False)

    print(f"Forces sauvegardé : {filename}")


def is_in_contact_with_wall(gr):
    return (
        gr.pos[0] - gr.radius <= 0 or
        gr.pos[0] + gr.radius >= 100 or
        gr.pos[1] - gr.radius <= 0 or
        gr.pos[1] + gr.radius >= 100
    )


def find_middle_grain():
    candidates = [
        gr for gr in dem.simu.grain_list
        if not is_in_contact_with_wall(gr) and gr not in gray_grain_list_down
    ]

    if not candidates:
        return None

    center = np.array([50., 50.])
    return min(candidates, key=lambda g: np.linalg.norm(g.pos - center))


def print_middle_grain_force_balance():
    gr = find_middle_grain()
    if gr is None:
        print("Aucun grain intérieur trouvé.")
        return

    Fx, Fy = gr.force[0], gr.force[1]
    Fnorm = np.linalg.norm(gr.force)

    print("\nGrain intérieur choisi")
    print(f"Position : ({gr.pos[0]:.3f}, {gr.pos[1]:.3f})")
    print(f"Rayon    : {gr.radius:.3f}")
    print(f"Somme des forces = ({Fx:.6f}, {Fy:.6f})")
    print(f"Norme de la résultante = {Fnorm:.6e}")


def is_system_at_equilibrium(threshold_velocity=0.5):
    total_velocity = sum(np.linalg.norm(gr.vel) for gr in dem.simu.grain_list)
    avg_velocity = total_velocity / len(dem.simu.grain_list)
    max_force = max(np.linalg.norm(gr.force) for gr in dem.simu.grain_list)
    print(avg_velocity)
    return avg_velocity < threshold_velocity


def time_loop():
    reset_force()
    add_gravity_force()
    add_dissipation()
    manage_contact_and_update_forces()
    add_mirror_grains_and_contacts()
    add_extern_force()

    calculate_total_force()
    rigid_wall()
    velocity_verlet()
    apply_boundary_condition()
    extrapoint()
    update_grain_colors()

    if is_system_at_equilibrium():
        print_middle_grain_force_balance()

        save_nodes(CURRENT_SIM_ID)
        save_forces(CURRENT_SIM_ID)

        filename_nuances = os.path.join("output", f"simu_nuances_{CURRENT_SIM_ID}.png")
        save_simulation_image(dem.simu.grain_list, filename_nuances, use_grain_color=True)
        return True

    return False


def find_next_simulation_id(output_dir="output"):
    """
    Cherche le prochain identifiant disponible à partir des fichiers nodes_*.csv
    Exemple :
    si nodes_0.csv ... nodes_999.csv existent déjà,
    la fonction renvoie 1000.
    """
    max_id = -1

    if not os.path.exists(output_dir):
        return 0

    for filename in os.listdir(output_dir):
        if filename.startswith("nodes_") and filename.endswith(".csv"):
            try:
                sim_id = int(filename[len("nodes_"):-len(".csv")])
                max_id = max(max_id, sim_id)
            except ValueError:
                pass

    return max_id + 1


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    total_target = 3000   # on veut aller jusqu'à 2999
    start_id = find_next_simulation_id("output")

    print(f"Reprise automatique à partir de la simulation {start_id}")
    print(f"Objectif total : {total_target} simulations")

    if start_id >= total_target:
        print("Les 3000 simulations sont déjà générées.")
    else:
        for simulation_id in range(start_id, total_target):
            print(f"\n===== Lancement simulation {simulation_id} =====")
            run_simulation(simulation_id)
