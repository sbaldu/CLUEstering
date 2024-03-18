
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(1, '../CLUEstering/')
import CLUEstering as clue


def run_blobs(backend: str) -> int:
    '''
    '''
    c = clue.clusterer(0.2, 5, 1.)
    c.read_data("../tests/test_datasets/blobs.csv")
    c.run_clue(backend)
    # c.cluster_plotter()

    return c.elapsed_time


def run_blobs_noise(backend: str) -> int:
    '''
    '''
    c = clue.clusterer(0.2, 5, 1.)
    c.read_data("../tests/test_datasets/blobs_noise.csv")
    c.run_clue(backend)
    # c.cluster_plotter()

    return c.elapsed_time


def run_blobs_3d(backend: str) -> int:
    '''
    '''
    c = clue.clusterer(0.8, 5, 1.)
    c.read_data("../tests/test_datasets/blobs_3d.csv")
    c.run_clue(backend)
    c.cluster_plotter()

    return c.elapsed_time


def run_blobs_3d_noise(backend: str) -> int:
    '''
    '''
    c = clue.clusterer(0.8, 5, 1.)
    c.read_data("../tests/test_datasets/blobs_3d_noise.csv")
    c.run_clue(backend)
    c.cluster_plotter()

    return c.elapsed_time


def run_moons(backend: str) -> int:
    '''
    '''
    c = clue.clusterer(0.5, 5., 1.)
    c.read_data("../tests/test_datasets/moons.csv")
    c.run_clue(backend)
    # c.cluster_plotter()

    return c.elapsed_time


def run_moons_3d(backend: str) -> int:
    '''
    '''
    c = clue.clusterer(0.6, 5., 2.)
    c.read_data("../tests/test_datasets/moon_3d.csv")
    c.run_clue(backend)
    c.cluster_plotter()

    return c.elapsed_time


def run_circles(backend: str) -> int:
    '''
    '''
    c = clue.clusterer(0.5, 5, 5.)
    c.read_data("../tests/test_datasets/circles.csv")
    c.run_clue(backend)
    # c.cluster_plotter()

    return c.elapsed_time


def run_circles_3d(backend: str) -> int:
    '''
    '''
    c = clue.clusterer(0.8, 5, 2.)
    c.read_data("../tests/test_datasets/circles_3d.csv")
    c.run_clue(backend)
    c.input_plotter()
    c.cluster_plotter()

    return c.elapsed_time


def run_toy_detector(backend: str) -> int:
    '''
    '''
    c = clue.clusterer(0.05, 5., 1.)
    c.read_data("../tests/test_datasets/toyDetector.csv")
    c.run_clue(backend)
    # c.cluster_plotter()

    return c.elapsed_time


def run_sissa(backend: str) -> int:
    '''
    '''
    c = clue.clusterer(0.2, 2., 1.)
    c.read_data("../tests/test_datasets/sissa.csv")
    c.run_clue(backend)
    # c.cluster_plotter()

    return c.elapsed_time


if __name__ == "__main__":
    runs_2d = {'datasets': ['blobs',
                            'blobs_noise',
                            'moons',
                            'circles',
                            'toy_detector',
                            'sissa'],
               'time_cpu': [run_blobs('cpu serial'),
                            run_blobs_noise('cpu serial'),
                            run_moons('cpu serial'),
                            # run_circles('cpu serial'),
                            run_toy_detector('cpu serial'),
                            run_sissa('cpu serial')],
               'time_tbb': [run_blobs('cpu tbb'),
                            run_blobs_noise('cpu tbb'),
                            run_moons('cpu tbb'),
                            # run_circles('cpu tbb'),
                            run_toy_detector('cpu tbb'),
                            run_sissa('cpu tbb')]}
    runs_3d = {'datasets': ['blobs',
                            'blobs_noise',
                            'moons',
                            'circles'],
               'time_cpu': [run_blobs_3d('cpu serial'),
                            run_blobs_3d_noise('cpu serial'),
                            # run_moons_3d('cpu serial'),
                            # run_circles_3d('cpu serial')],
                            run_moons_3d('cpu serial')],
               'time_tbb': [run_blobs_3d('cpu tbb'),
                            run_blobs_3d_noise('cpu tbb'),
                            # run_moons_3d('cpu tbb'),
                            # run_circles_3d('cpu tbb')]}
                            run_moons_3d('cpu tbb')]}
    df_2d = pd.DataFrame(runs_2d)
    df_3d = pd.DataFrame(runs_3d)

    width = 0.1
    df_2d.plot.bar(x='datasets', y=['time_cpu', 'time_tbb'],
                   color=['cornflowerblue', 'red'])
    plt.xlabel('Dataset')
    plt.ylabel('Execution time (ms)')
    plt.show()
    df_3d.plot.bar(x='datasets', y=['time_cpu', 'time_tbb'],
                   color=['cornflowerblue', 'red'])
    plt.show()
