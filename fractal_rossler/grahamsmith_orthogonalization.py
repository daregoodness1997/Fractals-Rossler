import numpy as np

class GrahamSmithOrthogonalization:
    def __init__(self, vectors):
        """
        Initialize the class with a list of vectors.
        
        :param vectors: A list of numpy arrays representing the vectors.
        """
        self.vectors = np.array(vectors)
        self.orthogonal_vectors = None
        self.orthonormal_vectors = None

    def orthogonalize(self):
        """
        Perform the Gram-Schmidt process to orthogonalize the input vectors.
        
        :return: A numpy array of orthogonal vectors.
        """
        num_vectors = self.vectors.shape[0]
        orthogonal_vectors = np.zeros_like(self.vectors)

        for i in range(num_vectors):
            v_i = self.vectors[i]
            # Start with the original vector
            orthogonal_vectors[i] = v_i
            
            # Subtract the projection of v_i onto all previously computed orthogonal vectors
            for j in range(i):
                proj = np.dot(v_i, orthogonal_vectors[j]) / np.dot(orthogonal_vectors[j], orthogonal_vectors[j]) * orthogonal_vectors[j]
                orthogonal_vectors[i] -= proj

        self.orthogonal_vectors = orthogonal_vectors
        return self.orthogonal_vectors

    def normalize(self):
        """
        Normalize the orthogonal vectors to create an orthonormal set.
        
        :return: A numpy array of orthonormal vectors.
        """
        if self.orthogonal_vectors is None:
            raise ValueError("You must first call the 'orthogonalize' method before normalizing.")
        
        num_vectors = self.orthogonal_vectors.shape[0]
        orthonormal_vectors = np.zeros_like(self.orthogonal_vectors)

        for i in range(num_vectors):
            norm = np.linalg.norm(self.orthogonal_vectors[i])
            if norm == 0:
                raise ValueError("Cannot normalize a zero vector.")
            orthonormal_vectors[i] = self.orthogonal_vectors[i] / norm

        self.orthonormal_vectors = orthonormal_vectors
        return self.orthonormal_vectors

    def get_orthogonal(self):
        """Return the orthogonal vectors."""
        if self.orthogonal_vectors is None:
            raise ValueError("Orthogonalization has not been performed yet.")
        return self.orthogonal_vectors

    def get_orthonormal(self):
        """Return the orthonormal vectors."""
        if self.orthonormal_vectors is None:
            raise ValueError("Normalization has not been performed yet.")
        return self.orthonormal_vectors
