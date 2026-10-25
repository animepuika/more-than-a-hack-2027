"""DFL open-data loader on a tiny synthetic match."""
from pathlib import Path
import tempfile
import unittest
import numpy as np

from solution.dfl_open import load_dfl_match

INFO = """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest><MatchInformation>
<General HomeTeamId="T-H" GuestTeamId="T-G" MatchId="DFL-MAT-TEST" />
<Environment PitchX="105.00" PitchY="68.00" />
<Teams><Team TeamId="T-G" Role="guest"/><Team TeamId="T-H" Role="home"/></Teams>
</MatchInformation></PutDataRequest>"""

def frameset(section, team, person, frames, ball=False):
    rows = []
    for n, x, y, *extra in frames:
        attrs = f'N="{n}" X="{x}" Y="{y}"'
        if ball:
            attrs += f' BallPossession="{extra[0]}" BallStatus="{extra[1]}"'
        rows.append(f'<Frame {attrs}/>')
    return f'<FrameSet GameSection="{section}" TeamId="{team}" PersonId="{person}">' + ''.join(rows) + '</FrameSet>'

POSITIONS = ('<?xml version="1.0" encoding="UTF-8"?><PutDataRequest><Positions>'
             + frameset('firstHalf', 'referee', 'R1', [(10000, 0, 0), (10001, 0, 0)])
             + frameset('firstHalf', 'T-H', 'P1', [(10000, -10, 5), (10001, -9, 5)])
             + frameset('firstHalf', 'T-G', 'P2', [(10001, 20, -4)])
             + frameset('firstHalf', 'BALL', 'B', [(10000, 0, 0, 1, 1), (10001, 1, 0, 2, 0)], ball=True)
             + frameset('secondHalf', 'T-H', 'P1', [(100000, 3, 3)])
             + frameset('secondHalf', 'BALL', 'B', [(100000, 3, 3, 2, 1)], ball=True)
             + '</Positions></PutDataRequest>')


class DflLoaderTests(unittest.TestCase):
    def test_labels_positions_and_halves(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder/'DFL_02_01_matchinformation_X_DFL-MAT-TEST.xml').write_text(INFO, encoding='utf-8')
            positions = folder/'DFL_04_03_positions_raw_observed_X_DFL-MAT-TEST.xml'
            positions.write_text(POSITIONS, encoding='utf-8')
            loaded = load_dfl_match(folder/'DFL_02_01_matchinformation_X_DFL-MAT-TEST.xml', positions, folder/'cache')
            again = load_dfl_match(folder/'DFL_02_01_matchinformation_X_DFL-MAT-TEST.xml', positions, folder/'cache')
        m = loaded.match
        self.assertEqual(''.join(loaded.labels), 'HDA')
        self.assertEqual(m.period.tolist(), [1, 1, 2])
        self.assertEqual(m.home_slots.tolist(), [False, True])   # slots sorted: T-G/P2, T-H/P1; referee skipped
        np.testing.assert_allclose(m.player_xy_m[0, 1], [42.5, 39.0])
        self.assertTrue(np.isnan(m.player_xy_m[0, 0]).all())     # guest player not yet on the pitch
        np.testing.assert_allclose(m.ball_xy_m[1], [53.5, 34.0])
        np.testing.assert_array_equal(again.labels, loaded.labels)  # cache round trip


if __name__ == '__main__':
    unittest.main()
