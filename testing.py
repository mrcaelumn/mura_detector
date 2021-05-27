from pprint import pprint
def numRookCaptures(board):
    """
    :type board: List[List[str]]
    :rtype: int
    """
    return sum([s.count('Rp') + s.count('pR') for s in
                [''.join(row).replace('.', '') for row in (board + list(map(list, zip(*board)))) if 'R' in row]])


if __name__ == '__main__':
    board = [[".", ".", ".", ".", ".", ".", ".", "."], [".", ".", ".", "p", ".", ".", ".", "."],
             [".", ".", ".", "R", ".", ".", ".", "p"], [".", ".", ".", ".", ".", ".", ".", "."],
             [".", ".", ".", ".", ".", ".", ".", "."], [".", ".", ".", "p", ".", ".", ".", "."],
             [".", ".", ".", ".", ".", ".", ".", "."], [".", ".", ".", ".", ".", ".", ".", "."]]

    # test = numRookCaptures(board)
    # print(test)
    number_list = [[1, 2, 3,4], [2, 3], [3, 4]]
    pprint(board+zip(*board))
