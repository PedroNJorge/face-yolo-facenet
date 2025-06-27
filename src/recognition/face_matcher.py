from torch.nn.functional import cosine_similarity


class FaceMatcher():
    def is_match(embedding1, embedding2, threshold=0.5):
        '''
        Check if the faces are the same with cosine_similarity
            -> -1 = Opposites | 0 = No Similarity | 1 = Identical

        Args:
            embeddingN (torch Tensor): Embedded Face
            threshold (float): Threshold that defines if the faces correspond
                                to the same person

        Returns:
            Boolean depending if score > threshold
        '''
        score = cosine_similarity(embedding1, embedding2)
        return score > threshold
