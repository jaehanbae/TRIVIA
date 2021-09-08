# TRIVIA: Tools foR Interactive fits VIsuAlization
Make AAS journal-compatible interactive figures from 3D fits cubes.

Don’t bother making/staring at pages of channel maps!

# Usage
To make interactive channel maps, simply use
```bash
from trivia import make_cm
make_cm('path_to_fits')
```
![cm_example](https://user-images.githubusercontent.com/42013416/126183507-070b1a65-b99a-45ae-b0c7-f2de7deff992.png)

To make a position-position-velocity diagram, simply use
```bash
from trivia import make_ppv
make_ppv('path_to_fits')
```
![ppv_example](https://user-images.githubusercontent.com/42013416/126183684-fb3b3a4c-8039-4f3a-9261-51d42c185444.png)



