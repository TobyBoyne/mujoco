$$ \text{extrude}(\mathbf{p}, \mathbf{g}, S, h) $$


```cpp
// compute the greatest distance, comparing the xy-plane to the z-plane
float gd = max(S, |z|-h)
```

```cpp
d/dx gd = S > |z|-h ? dS/dx : 0
d/dy gd = S > |z|-h ? dS/dy : 0
d/dz gd = S > |z|-h ? 0     : sign(z)
```

```cpp
d/dx min(gd, 0) = gd < 0 ? d/dx gd : 0
```

```cpp
S_0 = max(S, 0)
zh_0 = max(|z| - h, 0)
ell = length({S_0, zh_0})

// note that d/du length(u, v) = u/length(u, v)
```

```cpp
d/dx ell = S > 0 ? dS/dx * S_0/ell : 0
d/dy ell = S > 0 ? dS/dy * S_0/ell : 0
d/dz ell = |z|-h > 0 ? sign(z) * zh_0/ell : 0
```

