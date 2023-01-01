import os
import numpy as np
from wafer.logger import lg
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from wafer.entities.artifact import DataPreparationArtifact
from kneed import KneeLocator
from typing import Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ClusterDataInstances:
    """Divides the given data instances into clusters via KMeans Clustering algorithm.

    Args:
        X (np.array): Takes in an array which gotta be clustered.
        desc (str): Description of the said array.
        data_prep_config (DataPreparationArtifact): Config object from the Data Preparation stage to get the config for saving Elbow 
        plot. 
    """
    X: np.array
    desc: str
    data_prep_config: DataPreparationArtifact

    def _get_ideal_number_of_clusters(self):
        """Returns the ideal number of clusters the given data instances should be divided into by locating the dispersal
        point in number of clusters vs WCSS plot.

        Raises:
            e: Raises relevant exception should any kinda error pops up while determining the ideal number of clusters.

        Returns:
            int: Ideal number of clusters the given data instances should be divided into.
        """
        try:
            lg.info(
                f'Getting the ideal number of clusters to cluster "{self.desc} set" into..')

            ####################### Compute WCSS for shortlisted number of clusters ##########################
            lg.info("computing WCSS for shortlisted number of clusters..")
            wcss = []  # Within Summation of Squares
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i, init='k-means++',
                                random_state=42)
                kmeans.fit(self.X)
                wcss.append(kmeans.inertia_)
                lg.info(f"WCSS for n_clusters={i}: {kmeans.inertia_}")
            lg.info(
                "WCSS computed successfully for all shortlisted number of clusters!")

            ################################# Export Elbow plot ##############################################
            lg.info("Exporting the Elbow plot..")
            plt.plot(range(1, 11), wcss)
            plt.title("Elbow method")
            plt.xlabel("No. of Clusters")
            plt.ylabel("WCSS")
            # Make sure the dir where to export `Elbow plot` does exist
            elbow_plot_dir = os.path.dirname(
                self.data_prep_config.elbow_plot_path)
            os.makedirs(elbow_plot_dir, exist_ok=True)
            plt.savefig(self.data_prep_config.elbow_plot_path)
            lg.info(
                f'Elbow plot exported successfully to "{self.data_prep_config.elbow_plot_path}"')

            ################### Finalize dispersal point as the ideal number of clusters #####################
            lg.info(
                "Finding the ideal number of clusters (by locating the dispersal point) via Elbow method..")
            knee_finder = KneeLocator(
                range(1, 11), wcss, curve='convex', direction='decreasing')  # range(1, 11) vs WCSS
            lg.info(
                f"Ideal number of clusters to be formed: {knee_finder.knee}")

            return knee_finder.knee
            ...
        except Exception as e:
            lg.exception(e)
            raise e

    def create_clusters(self) -> Tuple:
        """Divides the given data instances into the different clusters, they first hand shoud've been divided into
        via offcourse Kmeans Clustering algorithm.

        Raises:
            e: Raises relevant exception should any kinda error pops up while dividing the given data instances into
            clusters.

        Returns:
            (KMeans, np.array): KMeans Clustering object being used to cluster the given data instances and the given dataset 
            along with the cluster labels, respectively.
        """
        try:
            ideal_clusters = self._get_ideal_number_of_clusters()
            lg.info(
                f"Dividing the \"{self.desc}\" instances into {ideal_clusters} clusters via KMeans Clustering algorithm..")
            kmeans = KMeans(n_clusters=ideal_clusters,
                            init='k-means++', random_state=42)
            y_kmeans = kmeans.fit_predict(self.X)
            lg.info(
                f"..said data instances divided into {ideal_clusters} clusters successfully!")

            return kmeans, np.c_[self.X, y_kmeans]
            ...
        except Exception as e:
            lg.exception(e)
            raise e
