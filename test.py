from direct.showbase.ShowBase import ShowBase
# imports Panda3D library for the 3D game window

from panda3d.core import (
    AmbientLight,
    BitMask32,
    CollisionBox,
    CollisionHandlerQueue,
    CollisionNode,
    CollisionRay,
    CollisionTraverser,
    ColorAttrib,
    DirectionalLight,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LColor,
    NodePath,
    PNMImage,
    Point3,
    Texture,
)

# Imports necessary functions for the 3D game window such as lighting,collision detection, ground detection, and more)

class MyScene(ShowBase):
    _BUILDING_COLOR = (0.29, 0.18, 0.10, 1.0)
    # Basic scene for 3D world using brown /gray building color as the default

    _GRASS_BLOCK_COLOR = (0.52, 0.52, 0.52, 1)
    _flat_gray_texture = None
    _flat_gray_texture_rgb = None
    _flat_color_tex_cache = {}
    # Flat gray for the two simple grass blocks (solid with no shading) - Basic map 


    @classmethod
    # Function uses data shared by the class instead of creating a new one each time)
    def _get_flat_color_texture(cls, rgba):
        """1×1 texture for arbitrary solid RGBA (replaces box noise)."""
        key = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3] if len(rgba) > 3 else 1.0))
        cached = cls._flat_color_tex_cache.get(key)
        if cached is not None:
            return cached
        # Creates a new texture of solid color instead of the default noise texture
        img = PNMImage(1, 1)
        img.set_xel(0, 0, key[0], key[1], key[2])
        tex = Texture("flat_rgb_%g_%g_%g" % (key[0], key[1], key[2]))
        tex.load(img)
        cls._flat_color_tex_cache[key] = tex
        return tex
    # Creates image with pixels making the image into texture

    @classmethod
    # Function uses data shared by the class instead of creating a new one each time)
    def _get_flat_gray_texture(cls):
        """1×1 RGB texture — replaces `models/box` noise so the block is one solid color."""
        rgb = cls._GRASS_BLOCK_COLOR[:3]
        if cls._flat_gray_texture is not None and cls._flat_gray_texture_rgb == rgb:
            return cls._flat_gray_texture
        img = PNMImage(1, 1)
        img.set_xel(0, 0, rgb[0], rgb[1], rgb[2])
        tex = Texture("grass_block_flat_gray")
        tex.load(img)
        cls._flat_gray_texture = tex
        cls._flat_gray_texture_rgb = rgb
        return tex
    # Creates a new texture of solid color instead of the default noise texture

    @staticmethod
    # Helper function for specific parts of the map (placing paths, buildings, and more)
    def _center_node_xy(node):
        """Shift node so its tight XY bounds are centered on world (0, 0)."""
        node.flattenLight()
        mn, mx = node.getTightBounds()
        cx = (mn.getX() + mx.getX()) * 0.5
        cy = (mn.getY() + mx.getY()) * 0.5
        node.setPos(node.getX() - cx, node.getY() - cy, node.getZ())
    #Centers the node so its tight XY bounds are centered on world (0, 0)

    def _place_path_strip(self, scale_xyz, center_xy, grass_top):
        """Black path strip with bottom flush to grass_top; returns (node, (pmin, pmax))."""
        sx, sy, sz = scale_xyz
        cx, cy = center_xy
        strip = self.loader.loadModel("models/box")
        strip.setScale(sx, sy, sz)
        strip.setColor(0.15, 0.15, 0.15, 1)
        strip.reparentTo(self.render)
        strip.setPos(0, 0, 0)
        strip.flattenLight()
        mn, mx = strip.getTightBounds()
        ox = (mn.getX() + mx.getX()) * 0.5
        oy = (mn.getY() + mx.getY()) * 0.5
        strip.setPos(cx - ox, cy - oy, 0)
        strip.flattenLight()
        mn, mx = strip.getTightBounds()
        strip.setZ(strip.getZ() + grass_top - mn.getZ())
        strip.flattenLight()
        pmin, pmax = strip.getTightBounds()
        return strip, (pmin, pmax)
    # Creating a black path strip with bottom flush to grass_top using cube model; returns (node, (pmin, pmax))
    
    def _place_building_block(self, ursina_scale, ursina_pos, grass_top, dy=0, dz=0):
        """Ursina-style (w,h,d) and (x,y,z) with y=up, z=depth → Panda z-up on grass_top.
        dy shifts along Panda Y (ground depth); dz lowers the whole block on Z."""
        w, h, d = ursina_scale
        ux, uy, uz = ursina_pos
        b = self.loader.loadModel("models/box")
        b.setScale(w, d, h)
        self._paint_box_flat_rgba(b, self._BUILDING_COLOR)
        b.reparentTo(self.render)
        b.setPos(ux, uz + dy, grass_top + uy + dz)
        b.flattenLight()
        return b
    # Creating a building block using X = width, Y = depth, Z = height (Converts ursina scale and position to Panda3D scale and position)

    def _place_building_cap_on(self, lower, ursina_scale, z_epsilon=0.02):
        """Stack a smaller block on top of `lower`, centered in X/Y on lower's bounds."""
        w, h, d = ursina_scale
        lower.flattenLight()
        lmn, lmx = lower.getTightBounds()
        lcx = (lmn.getX() + lmx.getX()) * 0.5
        lcy = (lmn.getY() + lmx.getY()) * 0.5
        roof_z = lmx.getZ() + z_epsilon

        cap = self.loader.loadModel("models/box")
        cap.setScale(w, d, h)
        self._paint_box_flat_rgba(cap, self._BUILDING_COLOR)
        cap.reparentTo(self.render)
        cap.setPos(0, 0, 0)
        cap.flattenLight()
        cmn, cmx = cap.getTightBounds()
        ox = (cmn.getX() + cmx.getX()) * 0.5
        oy = (cmn.getY() + cmx.getY()) * 0.5
        cz = roof_z - cmn.getZ()
        cap.setPos(lcx - ox, lcy - oy, cz)
        cap.flattenLight()
        return cap
    # Creating a smaller block on top of `lower`, centered in X/Y on lower's bounds using cube model

    def _paint_box_flat_rgba(self, bb, fc):
        """Solid flat color on `models/box` (replaces noise texture)."""
        if len(fc) < 4:
            fc = (fc[0], fc[1], fc[2], 1.0)
        flat_tex = self._get_flat_color_texture(fc)
        coll = bb.findAllTextures()
        if coll.getNumTextures() > 0:
            old = [coll.getTexture(i) for i in range(coll.getNumTextures())]
            for t in old:
                bb.replace_texture(t, flat_tex)
        else:
            bb.setTextureOff()
            bb.setAttrib(ColorAttrib.makeFlat(LColor(fc[0], fc[1], fc[2], fc[3])))
        bb.setMaterialOff()
        bb.setLightOff()
        bb.setShaderOff()
        bb.setColor(1, 1, 1, 1)
    # Creating a solid flat color on `models/box` (replaces noise texture)

    def _make_square_pyramid_np(self, base_w, base_d, height):
        """Square pyramid, base in XY with center at origin, apex toward +Z."""
        fmt = GeomVertexFormat.getV3()
        vdata = GeomVertexData("pyramid", fmt, Geom.UHStatic)
        vw = GeomVertexWriter(vdata, "vertex")
        hx, hy = base_w * 0.5, base_d * 0.5
        vw.addData3(0, 0, height)
        vw.addData3(-hx, -hy, 0)
        vw.addData3(hx, -hy, 0)
        vw.addData3(hx, hy, 0)
        vw.addData3(-hx, hy, 0)
        tris = GeomTriangles(Geom.UHStatic)
        for i in range(4):
            a, b = 1 + i, 1 + ((i + 1) % 4)
            tris.addVertices(0, a, b)
        geom = Geom(vdata)
        geom.addPrimitive(tris)
        gn = GeomNode("pyramid_roof")
        gn.addGeom(geom)
        np = NodePath(gn)
        np.setAttrib(ColorAttrib.makeFlat(LColor(0, 0, 0, 1)))
        np.setMaterialOff()
        np.setLightOff()
        np.setShaderOff()
        np.setTwoSided(True)
        return np
        # Creating a square pyramid using base_w, base_d, and height (Base in XY with center at origin, triangles pointing toward +Z to create a pyramid faces)


    def _attach_black_pyramid_roof_on_box(self, box_np, height_scale=0.5):
        """Place a black pyramid on top of box: base matches XY roof footprint, height ∝ min side."""
        box_np.flattenLight()
        pmn, pmx = box_np.getTightBounds()
        cx = 0.5 * (pmn.getX() + pmx.getX())
        cy = 0.5 * (pmn.getY() + pmx.getY())
        bw = max(pmx.getX() - pmn.getX(), 1e-4)
        bd = max(pmx.getY() - pmn.getY(), 1e-4)
        ph = min(bw, bd) * height_scale
        roof = self._make_square_pyramid_np(bw, bd, ph)
        roof.reparentTo(self.render)
        roof.setPos(cx, cy, pmx.getZ())

    # Measures building size in order to create a pyramid on top of it (Base matches XY roof footprint)

    def _place_garden_in_rect(self, fmin, vert_path_bounds, hp1_bounds, hp2_bounds, grass_top):
        """Plants + pebbles on west grass between horizontal paths — never on black strips."""
        pminv, _pmaxv = vert_path_bounds

        PATH_X_CLEAR = 1.1  # west of vertical path min-X (world)
        PATH_Y_CLEAR = 1.05  # beyond horizontal strips' Y extents
        LAWN_X_CLEAR = 1.1  # inside west world edge of grass

        hp1_y_hi = max(hp1_bounds[0].getY(), hp1_bounds[1].getY())
        hp2_y_lo = min(hp2_bounds[0].getY(), hp2_bounds[1].getY())
        mn_y = hp1_y_hi + PATH_Y_CLEAR
        mx_y = hp2_y_lo - PATH_Y_CLEAR
        mn_x = fmin.getX() + LAWN_X_CLEAR
        mx_x = pminv.getX() - PATH_X_CLEAR
    


        if mn_x >= mx_x - 1.2 or mn_y >= mx_y - 1.2:
            return

        pad_w, pad_e, pad_s, pad_n = 0.5, 0.85, 0.4, 0.4
        smn_x, smx_x = mn_x + pad_w, mx_x - pad_e
        smn_y, smx_y = mn_y + pad_s, mx_y - pad_n
        if smn_x >= smx_x - 0.25 or smn_y >= smx_y - 0.25:
            return

        plant_margin = 0.35
        ix = smn_x + plant_margin
        ax = smx_x - plant_margin
        iy = smn_y + plant_margin
        ay = smx_y - plant_margin
        if ix >= ax - 0.25 or iy >= ay - 0.25:
            return

        y_lo_safe = mn_y + 0.12
        y_hi_safe = mx_y - 0.12

        east_limit = pminv.getX() - PATH_X_CLEAR - 0.18
        
    # Main functoin creating garden in a rectangle between horizontal paths and vertical path 
    # Plants + pebbles on west grass between horizontal paths — never on black strips.
    # Horizontal paths cover all X at two Y bands; vertical path covers a narrow X band for all Y.
    # We build a safe rectangle from each strip's tight bounds, then inset further for the brown slab.


        def _clamp_xy(px, py, hx, hy):
            px = max(ix + hx, min(ax - hx, px))
            py = max(iy + hy, min(ay - hy, py))
            px = min(east_limit - hx, px)
            px = max(ix + hx, px)
            py = max(y_lo_safe + hy, min(y_hi_safe - hy, py))
            return px, py
        
        # Helper functoin for clamping the x and y coordinates to the edges of the garden to kep objeects inside the garden


        for k in range(22):
            u = ((k * 17) % 100) / 100.0
            v = ((k * 31) % 100) / 100.0
            px = ix + u * (ax - ix)
            py = iy + v * (ay - iy)
            px, py = _clamp_xy(px, py, 0.2, 0.18)
            gry = 0.38 + 0.04 * (k % 5)
            self._place_block_on_grass(
                (0.28 + 0.04 * (k % 3), 0.24 + 0.03 * (k % 4), 0.05),
                (px, py),
                grass_top,
                flat_color=(gry, gry, gry * 1.02, 1),
            )

        ncols = max(3, min(7, int((ax - ix) / 2.6)))
        nrows = max(5, min(11, int((ay - iy) / 2.35)))

        # Runs 22 times to create 22 pebbles in the garden, spreads out through percentage values --> actual coordinates, then places a pebble at each position, changing gray shade)

        for ri in range(nrows):
        # Loop for every plant row 
            for ci in range(ncols):
                px = ix + (ci + 0.5) / ncols * (ax - ix)
                py = iy + (ri + 0.5) / nrows * (ay - iy)
                jx = (((ri * 5 + ci) % 5) - 2) * 0.16
                jy = (((ri * 3 + ci * 7) % 5) - 2) * 0.16
                px += jx
                py += jy
                is_bush = (ri * 13 + ci * 7) % 11 == 0
            
            # Loop for every plant column (puts plants in a grid pattern center, creaing plant offset location, certain grid locations designated for bushes)

                if is_bush:
                    px, py = _clamp_xy(px, py, 0.45, 0.45)
                    bush = self._place_block_on_grass(
                        (0.75, 0.75, 0.52),
                        (px, py),
                        grass_top,
                        flat_color=(0.1, 0.44, 0.13, 1),
                    )
                    self._place_block_on_grass(
                        (0.82, 0.82, 0.08),
                        (px, py),
                        grass_top,
                        stack_on=bush,
                        flat_color=(0.14, 0.52, 0.16, 1),
                    )
                    continue

                px, py = _clamp_xy(px, py, 0.26, 0.26)
                h_stalk = 0.38 + 0.08 * ((ri + ci * 2) % 6)
                g = 0.44 + 0.03 * ((ri * 11 + ci) % 8)
                stalk = self._place_block_on_grass(
                    (0.22, 0.22, h_stalk),
                    (px, py),
                    grass_top,
                    flat_color=(0.1, g, 0.12, 1),
                )
                top = stalk

                # If bush creates bush instead of plant, places a bush at the location (Having larger grid locations designated for bushes, places flat block on top of green bush)
                # Keeping plant inside the garden by clamping the x and y coordinates to the edges of the garden to kep objeects inside the garden
                # Plants different heights based on row and column position, changing green shade to create variety)

                if (ri + ci) % 2 == 0:
                    top = self._place_block_on_grass(
                        (0.48, 0.48, 0.07),
                        (px, py),
                        grass_top,
                        stack_on=stalk,
                        flat_color=(0.12, min(0.58, g + 0.08), 0.14, 1),
                    )
                # Depending on grid position, plants get flatter leaves block on top of the plant block)
                

                if (ri + ci * 3) % 5 == 0:
                    hues = (
                        (0.82, 0.22, 0.24, 1),
                        (0.92, 0.72, 0.18, 1),
                        (0.72, 0.32, 0.58, 1),
                        (0.95, 0.55, 0.2, 1),
                    )
                    self._place_block_on_grass(
                        (0.14, 0.14, 0.12),
                        (px, py),
                        grass_top,
                        stack_on=top,
                        flat_color=hues[(ri + ci) % 4],
                    )
                # Places a colorful flower at the location (Having certain grid locations designated for flowers, places flat block on top of the plant block)

                elif (ri + ci * 5) % 7 == 1:
                    self._place_block_on_grass(
                        (0.12, 0.12, 0.1),
                        (px, py),
                        grass_top,
                        stack_on=top,
                        flat_color=(0.95, 0.92, 0.88, 1),
                    )

                # If plant did not get colorful flower, places a white flower at the location (Having certain grid locations designated for flowers, places flat block on top of the plant block)

  


    def _place_block_on_grass(
        self, scale_xyz, center_xy, grass_top, solid_gray=False, stack_on=None, flat_color=None
    ):
        """Simple box on grass, or stacked on another node (bottom flush to stack_on roof).
        flat_color=(r,g,b[,a]) → solid flat that color; elif solid_gray → grass gray;
        else → solid brown (_BUILDING_COLOR)."""
        sx, sy, sz = scale_xyz
        cx, cy = center_xy
        bb = self.loader.loadModel("models/box")
        bb.setScale(sx, sy, sz)
        bb.reparentTo(self.render)
        if flat_color is not None:
            self._paint_box_flat_rgba(bb, flat_color)
        elif solid_gray:
            # Panda's models/box uses maps/noise.rgb; texture-off on the root does not strip it.
            gray_tex = self._get_flat_gray_texture()
            coll = bb.findAllTextures()
            if coll.getNumTextures() > 0:
                old = [coll.getTexture(i) for i in range(coll.getNumTextures())]
                for t in old:
                    bb.replace_texture(t, gray_tex)
            else:
                bb.setTextureOff()
                c = self._GRASS_BLOCK_COLOR
                bb.setAttrib(ColorAttrib.makeFlat(LColor(c[0], c[1], c[2], c[3])))
            bb.setMaterialOff()
            bb.setLightOff()
            bb.setShaderOff()
            bb.setColor(1, 1, 1, 1)
        else:
            self._paint_box_flat_rgba(bb, self._BUILDING_COLOR)
        bb.setPos(cx, cy, 0)
        bb.flattenLight()
        mn, mx = bb.getTightBounds()
        foot = grass_top
        if stack_on is not None:
            stack_on.flattenLight()
            _, smx = stack_on.getTightBounds()
            foot = smx.getZ() + 0.02
        bb.setZ(bb.getZ() + foot - mn.getZ())
        bb.flattenLight()
        mn, mx = bb.getTightBounds()
        if mn.getZ() > foot:
            bb.setZ(bb.getZ() - (mn.getZ() - foot))
        elif mn.getZ() < foot - 1e-4:
            bb.setZ(bb.getZ() + (foot - mn.getZ()))
        bb.flattenLight()
        return bb

          # Main function for placing boxes (scaling, centering, colroing, alignining with ground )
          # Simple box on grass, or stacked on another node (bottom flush to stack_on roof).
          # flat_color=(r,g,b[,a]) → solid flat that color; elif solid_gray → grass gray;
       # else → solid brown (_BUILDING_COLOR).


    def _stack_block_centered(
        self, scale_xyz, stack_on, cx, cy, grass_top, solid_gray=False, flat_color=None
    ):
        """Stack a block on `stack_on` and align its XY tight-bounds center to (cx, cy)."""
        node = self._place_block_on_grass(
            scale_xyz,
            (cx, cy),
            grass_top,
            solid_gray=solid_gray,
            stack_on=stack_on,
            flat_color=flat_color,
        )
        node.flattenLight()
        mn, mx = node.getTightBounds()
        ncx = 0.5 * (mn.getX() + mx.getX())
        ncy = 0.5 * (mn.getY() + mx.getY())
        node.setPos(node.getX() + cx - ncx, node.getY() + cy - ncy, node.getZ())
        node.flattenLight()
        return node

        # Stacks a block on another node and aligns its XY tight-bounds center to (cx, cy)

    def _nudge_block_xy_inside(self, slab, mn_x_lo, mx_x_hi, mn_y_lo, mx_y_hi):
        """Shift slab in X/Y until getTightBounds() lies inside the given axis-aligned limits."""
        for _ in range(16):
            slab.flattenLight()
            mn, mx = slab.getTightBounds()
            dx = dy = 0.0
            if mn.getX() < mn_x_lo:
                dx += mn_x_lo - mn.getX()
            if mx.getX() > mx_x_hi:
                dx += mx_x_hi - mx.getX()
            if mn.getY() < mn_y_lo:
                dy += mn_y_lo - mn.getY()
            if mx.getY() > mx_y_hi:
                dy += mx_y_hi - mx.getY()
            if abs(dx) < 1e-7 and abs(dy) < 1e-7:
                break
            slab.setPos(slab.getX() + dx, slab.getY() + dy, slab.getZ())

    # Shifts slab in X/Y until getTightBounds() lies inside the given axis-aligned limits.

    def _nudge_rig_clear_paths_xy(self, rig, fmin, fmax, path_bounds, grass_top, pad=0.14):
        """Slide rig in XY until its tight bounds avoid all path strips and stay on grass; then Z to grass."""
        for _ in range(64):
            rig.flattenLight()
            rmn, rmx = rig.getTightBounds()
            rx0, ry0 = rmn.getX(), rmn.getY()
            rx1, ry1 = rmx.getX(), rmx.getY()
            best = None
            for pmn, pmx in path_bounds:
                px0, py0 = pmn.getX() - pad, pmn.getY() - pad
                px1, py1 = pmx.getX() + pad, pmx.getY() + pad
                if rx1 <= px0 or rx0 >= px1 or ry1 <= py0 or ry0 >= py1:
                    continue
                opts = (
                    (px0 - rx1, 0.0),
                    (px1 - rx0, 0.0),
                    (0.0, py0 - ry1),
                    (0.0, py1 - ry0),
                )
                for ox, oy in opts:
                    s = abs(ox) + abs(oy)
                    if s < 1e-8:
                        continue
                    if best is None or s < best[0]:
                        best = (s, ox, oy)
            if best is not None:
                rig.setPos(rig.getX() + best[1], rig.getY() + best[2], rig.getZ())
                continue
            dx = dy = 0.0
            if rmn.getX() < fmin.getX() + pad:
                dx += fmin.getX() + pad - rmn.getX()
            if rmx.getX() > fmax.getX() - pad:
                dx += fmax.getX() - pad - rmx.getX()
            if rmn.getY() < fmin.getY() + pad:
                dy += fmin.getY() + pad - rmn.getY()
            if rmx.getY() > fmax.getY() - pad:
                dy += fmax.getY() - pad - rmx.getY()
            if abs(dx) < 1e-7 and abs(dy) < 1e-7:
                break
            rig.setPos(rig.getX() + dx, rig.getY() + dy, rig.getZ())
        rig.flattenLight()
        tmn, _ = rig.getTightBounds()
        rig.setZ(rig.getZ() + grass_top - tmn.getZ())
        rig.flattenLight()

    # Slides rig in XY until its tight bounds avoid all path strips and stay on grass; then Z to grass.

    def _shove_rig_to_cell_corner_then_clear(self, rig, corner, fmin, fmax, path_bounds, grass_top, cp=0.32):
        """Push rig toward (+X, low Y) corner of grass cell `corner`, then clear paths / grass edge."""
        x_lo, x_hi, y_lo, y_hi = corner
        for _ in range(12):
            rig.flattenLight()
            rmn, rmx = rig.getTightBounds()
            tx_hi = min(x_hi, fmax.getX()) - cp
            ty_lo = max(y_lo, fmin.getY()) + cp
            dx = tx_hi - rmx.getX()
            dy = ty_lo - rmn.getY()
            if abs(dx) < 0.04 and abs(dy) < 0.04:
                break
            rig.setPos(rig.getX() + dx, rig.getY() + dy, rig.getZ())
            self._nudge_rig_clear_paths_xy(rig, fmin, fmax, path_bounds, grass_top, pad=0.16)
    # Pushes rig toward (+X, low Y) corner of grass cell `corner`, then clear paths / grass edge.

    def _place_gray_slab_tower(
        self,
        mn_x_lo,
        mx_x_hi,
        mn_y_lo,
        mx_y_hi,
        grass_top,
        bias_high_y,
        lay_on_side=False,
        fmin=None,
        fmax=None,
        path_bounds=None,
        corner_cell=None,
    ):
        """Gray base + 6×(black small, gray large) stack; bias corner within the XY legal rect.
        lay_on_side: lay tower on its side (stack horizontal) and orient stripes along −Y (forward)."""
        slab_ix = (mn_x_lo + mx_x_hi) * 0.5
        slab_iy = (mn_y_lo + mx_y_hi) * 0.5
        slab = self._place_block_on_grass((16, 16, 1.45), (slab_ix, slab_iy), grass_top, solid_gray=True)
        self._nudge_block_xy_inside(slab, mn_x_lo, mx_x_hi, mn_y_lo, mx_y_hi)
        slab.flattenLight()
        mn, mx = slab.getTightBounds()
        cx, cy = 0.5 * (mn.getX() + mx.getX()), 0.5 * (mn.getY() + mx.getY())
        hw, hh = 0.5 * (mx.getX() - mn.getX()), 0.5 * (mx.getY() - mn.getY())
        cx_right = mx_x_hi - hw - 0.02
        if bias_high_y:
            cy_tgt = mn_y_lo + hh + 0.06 + 4.2
        else:
            cy_tgt = mn_y_lo + hh + 0.06 + 1.2
        slab.setPos(slab.getX() + cx_right - cx, slab.getY() + cy_tgt - cy, slab.getZ())
        self._nudge_block_xy_inside(slab, mn_x_lo, mx_x_hi, mn_y_lo, mx_y_hi)
        slab.flattenLight()
        smn, smx = slab.getTightBounds()
        cx = 0.5 * (smn.getX() + smx.getX())
        cy = 0.5 * (smn.getY() + smx.getY())
        prev = slab
        nodes = [slab]
        for _ in range(6):
            prev = self._stack_block_centered(
                (12.5, 12.5, 1.45), prev, cx, cy, grass_top, flat_color=(0, 0, 0, 1)
            )
            nodes.append(prev)
            prev.flattenLight()
            pmn, pmx = prev.getTightBounds()
            cx, cy = 0.5 * (pmn.getX() + pmx.getX()), 0.5 * (pmn.getY() + pmx.getY())
            prev = self._stack_block_centered(
                (16, 16, 1.45), prev, cx, cy, grass_top, solid_gray=True
            )
            nodes.append(prev)
            prev.flattenLight()
            pmn, pmx = prev.getTightBounds()
            cx, cy = 0.5 * (pmn.getX() + pmx.getX()), 0.5 * (pmn.getY() + pmx.getY())

        if lay_on_side:
            slab.flattenLight()
            bmn, bmx = slab.getTightBounds()
            px = 0.5 * (bmn.getX() + bmx.getX())
            py = 0.5 * (bmn.getY() + bmx.getY())
            pz = bmn.getZ()
            rig = self.render.attachNewNode("gray_slab_tower_side")
            rig.setPos(px, py, pz)
            for n in nodes:
                n.wrtReparentTo(rig)
            # Lay stack horizontal, then spin so stripes run along −Y (forward / into the map)
            rig.setHpr(-90, -90, 0)
            rig.flattenLight()
            tmn, _ = rig.getTightBounds()
            rig.setZ(rig.getZ() + grass_top - tmn.getZ())
            rig.flattenLight()
            # 45° left in plan view (about world Z)
            rig.setH(rig.getH() + 45.0)
            rig.flattenLight()
            tmn, _ = rig.getTightBounds()
            rig.setZ(rig.getZ() + grass_top - tmn.getZ())
            rig.flattenLight()
            if fmin is not None and fmax is not None and path_bounds is not None:
                self._nudge_rig_clear_paths_xy(rig, fmin, fmax, path_bounds, grass_top)
            if corner_cell is not None and fmin is not None and path_bounds is not None:
                self._shove_rig_to_cell_corner_then_clear(rig, corner_cell, fmin, fmax, path_bounds, grass_top)
    # Creates slab tower base + 6×(black small, gray large) stack using loops; bias corner within the XY legal rect, rotate 45° left in plan view (about world Z) to create a striped pattern)
   
    def __init__(self):
        ShowBase.__init__(self)
        self.disableMouse()
    # Runs the main function to create the map, turns off mouse cursor)

        self.win.setClearColor((0.5, 0.7, 1.0, 1))
        # Sky - sets the background color to a light blue

        floor = self.loader.loadModel("models/box")
        floor.setScale(52, 100, 0.5)
        floor.setPos(0, 0, -2)
        floor.setColor(0.3, 0.7, 0.3, 1)
        floor.reparentTo(self.render)
        self._center_node_xy(floor)
        floor.flattenLight()
        fmin, fmax = floor.getTightBounds()
        grass_top = fmax.getZ()
        # Floor — scale first; then center geometry on (0,0) so grass has a true world center (making large and flat and green)

        path, (pmin2, pmax2) = self._place_path_strip((8, 100, 0.08), (0, 0), grass_top)

        fy = fmax.getY() - fmin.getY()
        y_cross_a = fmin.getY() + fy / 3.0
        y_cross_b = fmin.getY() + 2.0 * fy / 3.0
        _hp1, hp1_bounds = self._place_path_strip((52, 8, 0.08), (0, y_cross_a), grass_top)
        _hp2, hp2_bounds = self._place_path_strip((52, 8, 0.08), (0, y_cross_b), grass_top)
        # Main black verticle path + two cross paths in horizontal positions, spaced evenly spaced on original path.


        _b_dy, _b_dz = 25.0, -12.8
        tower_base = self._place_building_block(
            (10, 20, 10), (10, 10.08, -30), grass_top, dy=_b_dy, dz=_b_dz
        )
        tower_cap = self._place_building_cap_on(tower_base, (7.5, 2.5, 7.5))
        tower_cap2 = self._place_building_cap_on(tower_cap, (5.0, 1.6, 5.0))
        self._place_building_cap_on(tower_cap2, (0.19, 3.75, 0.19))
        # Creating large gray tower and stacking smaller blocks / capson top of it to create a tower, adding a smalller and taller block / cap on the top)

        near_spawn_y = pmax2.getY() - 1.5
        path_left_x = pmin2.getX()
        inner_edge_x = path_left_x - 5.0 + 26.0
        half_x_wide, half_x_narrow = 7.0, 4.0
        block_x1 = inner_edge_x - half_x_wide - 3.0
        block_x2 = inner_edge_x - half_x_narrow
        block_y1 = near_spawn_y - 24.0
        block_y2 = block_y1 + 13.5
        self._place_block_on_grass((14, 8, 11), (block_x1, block_y1), grass_top, solid_gray=True)
        brown_near_path = self._place_block_on_grass(
            (8, 8, 11), (block_x2, block_y2), grass_top, solid_gray=False
        )
        self._attach_black_pyramid_roof_on_box(brown_near_path)
        # Calculating the position of the blocks and placing them on the grass, creating a brown block / building and black block / building near the path, adding a pyramid roof on the black block / building)



        gap = 0.55
        y_clear_above_cross = 0.22
        vx_lo = pmin2.getX()
        h2_top = hp2_bounds[1].getY()
        mn_x_lo = fmin.getX() + gap
        mx_x_hi = vx_lo - gap
        mn_y_lo = h2_top + y_clear_above_cross
        mx_y_hi = fmax.getY() - gap
        self._place_gray_slab_tower(mn_x_lo, mx_x_hi, mn_y_lo, mx_y_hi, grass_top, bias_high_y=True)

        # Copy of same slab tower — bottom-right grass cell (+X, low Y)



        vx_hi = pmax2.getX()
        h1_south = hp1_bounds[0].getY()
        br_mn_x_lo = vx_hi + gap
        br_mx_x_hi = fmax.getX() - gap
        br_mn_y_lo = fmin.getY() + gap
        br_mx_y_hi = h1_south - gap
        self._place_gray_slab_tower(
            br_mn_x_lo,
            br_mx_x_hi,
            br_mn_y_lo,
            br_mx_y_hi,
            grass_top,
            bias_high_y=False,
            lay_on_side=True,
            fmin=fmin,
            fmax=fmax,
            path_bounds=((pmin2, pmax2), hp1_bounds, hp2_bounds),
            corner_cell=(br_mn_x_lo, br_mx_x_hi, br_mn_y_lo, br_mx_y_hi),
        )
       # Calculates bottom-right grass area nand places a gray slab tower in it + roating sideways to create a striped pattern)


        vx_lo = pmin2.getX()
        h1_bot = hp1_bounds[0].getY()
        g_edge = 0.75
        tr_w, tr_d, tr_h = 11.0, 14.0, 7.0
        hw, hd = tr_w * 0.5, tr_d * 0.5
        tr_cx_lo = fmin.getX() + g_edge + hw
        tr_cx_hi = vx_lo - g_edge - hw
        tr_cy_lo = fmin.getY() + g_edge + hd
        tr_cy_hi = h1_bot - g_edge - hd
        tr_cx = max(tr_cx_lo, min(tr_cx_hi, tr_cx_lo + 1.8))
        tr_cy = max(tr_cy_lo, min(tr_cy_hi, tr_cy_lo + 1.5))
        brown_corner_block = self._place_block_on_grass((tr_w, tr_d, tr_h), (tr_cx, tr_cy), grass_top)
        self._attach_black_pyramid_roof_on_box(brown_corner_block)
        # Calculates bottom-left grass area and places a brown block / building, adding a black pyramid roof on the block / building)

        self._place_garden_in_rect(fmin, (pmin2, pmax2), hp1_bounds, hp2_bounds, grass_top)
        # Greating garden (plants, bushes, flowers, pubbles)


        pad = 0.08
        ground = CollisionNode("ground")
        ground.addSolid(
            CollisionBox(
                Point3(fmin.getX() - pad, fmin.getY() - pad, fmin.getZ() - pad),
                Point3(fmax.getX() + pad, fmax.getY() + pad, fmax.getZ() + pad),
            )
        )
        for mn, mx in ((pmin2, pmax2), hp1_bounds, hp2_bounds):
            ground.addSolid(
                CollisionBox(
                    Point3(mn.getX() - pad, mn.getY() - pad, mn.getZ() - pad),
                    Point3(mx.getX() + pad, mx.getY() + pad, mx.getZ() + pad),
                )
            )
        ground.setIntoCollideMask(BitMask32.bit(0))
        self.render.attachNewNode(ground)

        ray = CollisionRay()
        ray.setOrigin(0, 0, 50)
        ray.setDirection(0, 0, -1)
        ray_cn = CollisionNode("ground_ray")
        ray_cn.addSolid(ray)
        ray_cn.setFromCollideMask(BitMask32.bit(0))
        ray_cn.setIntoCollideMask(BitMask32.allOff())
        self._ground_ray_np = self.render.attachNewNode(ray_cn)
        self._ground_queue = CollisionHandlerQueue()
        self.cTrav = CollisionTraverser()
        self.cTrav.addCollider(self._ground_ray_np, self._ground_queue)
        self._eye_clearance = 3.75

        # Creates collision detection boxes for the floor and the paths to be able to walk on the grass and path

        px = (pmin2.getX() + pmax2.getX()) * 0.5
        spawn_y = pmax2.getY() - 1.5
        spawn_z = max(grass_top, pmax2.getZ()) + self._eye_clearance
        self.camera.setPos(px, spawn_y, spawn_z)
        self.camera.setHpr(180, 0, 0)
        # Spawn at the far (+Y) end of the black path, facing back along the strip

        alight = AmbientLight("ambient")
        alight.setColor((0.52, 0.52, 0.55, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)

        dlight = DirectionalLight("dlight")
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(0, -60, 0)
        self.render.setLight(dlnp)
        #Light — directional + ambient (brighter fill so tower reads as light gray)
        # Creates a directional light and an ambient light to light up the scene

        self.heading = 180
        self.pitch = 0
        self._mouse_ready = False
        self.win.requestProperties(self.win.getProperties())
        self.taskMgr.add(self.update, "update")
        # Prepares the mouse and starts continuous game loop to run the game 

        self.keys = {"w": False, "s": False, "a": False, "d": False}
        self.accept("w", self.setKey, ["w", True])
        self.accept("w-up", self.setKey, ["w", False])
        self.accept("s", self.setKey, ["s", True])
        self.accept("s-up", self.setKey, ["s", False])
        self.accept("a", self.setKey, ["a", True])
        self.accept("a-up", self.setKey, ["a", False])
        self.accept("d", self.setKey, ["d", True])
        self.accept("d-up", self.setKey, ["d", False])
        self.accept("escape", self.userExit)
        
        # Uses the WASD keyboard inputs to move the camera / around the scene, having escape key close the game


    def setKey(self, key, value):
        self.keys[key] = value
    # Sets the key to the value True or False, depending on which key is being pressed or released

    def update(self, task):
        dt = globalClock.getDt()
        speed = 24
    # Updates the function to run every at every frame, calculating the time between frames and setting the player speed to 24

        if self.keys["w"]: self.camera.setY(self.camera, speed * dt)
        if self.keys["s"]: self.camera.setY(self.camera, -speed * dt)
        if self.keys["a"]: self.camera.setX(self.camera, -speed * dt)
        if self.keys["d"]: self.camera.setX(self.camera, speed * dt)
        # Keyboard movement to move the camera forward, backward, left, and right


        cx, cy = self.win.getXSize() // 2, self.win.getYSize() // 2
        md = self.win.getPointer(0)
        if not self._mouse_ready:
            self.win.movePointer(0, cx, cy)
            self._mouse_ready = True
        elif self.win.movePointer(0, cx, cy):
            self.heading -= (md.getX() - cx) * 0.2
            self.pitch = max(-80, min(80, self.pitch - (md.getY() - cy) * 0.2))

        self.camera.setHpr(self.heading, self.pitch, 0)

        p = self.camera.getPos()
        self._ground_ray_np.setPos(p.getX(), p.getY(), p.getZ() + 80)
        self._ground_queue.clearEntries()
        self.cTrav.traverse(self.render)
        if self._ground_queue.getNumEntries() > 0:
            self._ground_queue.sortEntries()
            hit_z = self._ground_queue.getEntry(0).getSurfacePoint(self.render).getZ()
            self.camera.setZ(hit_z + self._eye_clearance)

        return task.cont
        
        # Rotating the player's view based on the mouse movement, detecting collisions with the ground and adjusting the camera height to the ground level

app = MyScene()
app.run()
# Runs the game window and starts the game loop