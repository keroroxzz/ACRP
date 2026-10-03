"""Selected single-image pipeline; rebuild all scene-dependent data per render.

No history, scene-name rules, CPU mask readback, or stored reference pictures.
"""
from cascade import *
from optimized_sources import shared_sources, lookup_sources, candidate_sources, bounded_source


class ScalarSurface:
    def __init__(self,width,height):
        self.w,self.h=width,height
        self.fbo=glGenFramebuffers(1)
        self.textures=[glGenTextures(1)]
        glBindTexture(GL_TEXTURE_2D,self.textures[0])
        glTexStorage2D(GL_TEXTURE_2D,1,GL_R32F,width,height)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST)
        glBindFramebuffer(GL_FRAMEBUFFER,self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,self.textures[0],0)
        if glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Incomplete scalar surface')

    bind=Surface.bind

    def close(self):
        glDeleteFramebuffers(1,[self.fbo])
        glDeleteTextures(self.textures)


class SingleFrameRenderer(Renderer):
    # At small resolutions fixed pass costs dominate. Route directly to the
    # original one-pass renderer, before any prepass is submitted.
    MIN_PIXELS=800*450

    def __init__(self,width,height):
        super().__init__(width,height)
        self.original_sky=self.sky
        self.original_reconstruct=self.reconstruct
        shared,probe=shared_sources()
        lookup,gather,event=lookup_sources(shared)
        candidate,selection=candidate_sources(lookup)
        self.optimized_source=bounded_source(candidate)
        self.optimized_sky=compile_program(self.optimized_source)
        self.prepasses=[
            (Surface(2,1),compile_program(probe),'probes',3,'uProbes'),
            (Surface(512,256),compile_program(gather),'gather',4,'uGather'),
            (Surface(64,32),compile_program(event),'events',5,'uEvents'),
            (ScalarSurface(512,512),compile_program(selection),'candidates',7,'uCandidates')]
        self.parent_score=compile_program((HERE/'parent_score.frag').read_text())
        self.scored_reconstruct=compile_program((HERE/'scored_reconstruct.frag').read_text())
        self.scores={}
        self.optimizing=False
        self.sky_only=False

    def scene(self,**kwargs):
        self.sky_only=bool(kwargs.get('sky_only',False))
        for p in [self.original_sky,self.optimized_sky]+[p[1] for p in self.prepasses]:
            self.sky=p
            super().scene(**kwargs)
        self.sky=self.original_sky

    def before_reconstruct(self,previous,current,threshold,radius,force):
        if not self.optimizing:
            return
        size=(previous.w+1,previous.h+1)
        glActiveTexture(GL_TEXTURE10)
        if size not in self.scores:
            self.scores[size]=ScalarSurface(*size)
        surface=self.scores[size]
        surface.bind()
        glDisable(GL_STENCIL_TEST)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,previous.textures[0])
        glUseProgram(self.parent_score)
        glUniform1i(self.loc(self.parent_score,'uPrevious'),0)
        glUniform1i(self.loc(self.parent_score,'uRadius'),radius)
        self.stage='score-%dx%d'%size
        self.triangle()
        glActiveTexture(GL_TEXTURE10)
        glBindTexture(GL_TEXTURE_2D,surface.textures[0])
        glUseProgram(self.scored_reconstruct)
        glUniform1i(self.loc(self.scored_reconstruct,'uScores'),10)

    def render(self,optimized=False,**variant):
        if optimized:
            variant=dict(levels=variant.pop('levels',2),factor=variant.pop('factor',4),
                         threshold=variant.pop('threshold',.035),**variant)
            if self.sky_only:
                # Procedural stars/disks are high frequency even when the
                # atmosphere is smooth. Keep all pixels for the sky-only path.
                variant=dict(levels=1)
        self.optimizing=optimized and self.w*self.h>=self.MIN_PIXELS
        self.sky=self.optimized_sky if self.optimizing else self.original_sky
        self.reconstruct=self.scored_reconstruct if self.optimizing else self.original_reconstruct
        if optimized and not self.optimizing:
            return super().render(levels=1)
        if self.optimizing:
            glDisable(GL_STENCIL_TEST)
            glColorMask(True,True,True,True)
            for surface,p,label,unit,name in self.prepasses:
                if self.sky_only and label!='gather':
                    continue
                surface.bind()
                glDrawBuffers(1,[GL_COLOR_ATTACHMENT0])
                glUseProgram(p)
                if label=='candidates':
                    glUniform1i(self.loc(p,'uEvents'),5)
                self.stage=label
                self.triangle()
                glActiveTexture(GL_TEXTURE0+unit)
                glBindTexture(GL_TEXTURE_2D,surface.textures[0])
                glUseProgram(self.sky)
                glUniform1i(self.loc(self.sky,name),unit)
        return self.render_levels(**variant)

    def render_levels(self,**variant):
        return super().render(**variant)

    def capture(self,**variant):
        if variant.get('optimized') and self.w*self.h<self.MIN_PIXELS:
            img,work=super().capture(levels=1)
            work['small_resolution_fallback']=True
            return img,work
        if variant.get('optimized') and self.sky_only:
            img,work=super().capture(optimized=True,levels=1)
            work['full_resolution_sky']=True
            return img,work
        if variant.get('optimized'):
            variant=dict(levels=variant.pop('levels',2),factor=variant.pop('factor',4),
                         threshold=variant.pop('threshold',.035),**variant)
        return super().capture(**variant)

    def profile(self,**variant):
        count=64
        self.profile_pool=[int(q) for q in np.asarray(glGenQueries(count)).reshape(-1)]
        self.profile_pending=[]
        self.render(**variant)
        glFinish()
        result={name:float(glGetQueryObjectuiv(q,GL_QUERY_RESULT))/1e6
                for name,q in self.profile_pending}
        self.profile_pending=None
        glDeleteQueries(count,self.profile_pool)
        return result

    def close(self):
        self.sky=self.original_sky
        self.reconstruct=self.original_reconstruct
        for surface,p,*_ in self.prepasses:
            surface.close()
            glDeleteProgram(p)
        for surface in self.scores.values():
            surface.close()
        for p in (self.optimized_sky,self.parent_score,self.scored_reconstruct):
            glDeleteProgram(p)
        super().close()
