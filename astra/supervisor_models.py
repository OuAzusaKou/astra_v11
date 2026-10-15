"""Model selection shared by local runner and laptop bridge."""
CHOICES=('astra','opus5.5','gemini','glm5.3-flash',
         'claude-opus-5-5','gemini-3.8-flash','glm-5.3-flash','qwen3.8-max')


def model_info(model):
    if model=='astra':
        return dict(provider='codex',model='gpt-6-astra',conversation_kind='codex_thread')
    from .api_supervisor import resolve_model
    info=dict(provider='quickrouter',model=resolve_model(model),conversation_kind='local_api_history')
    if info['model']=='qwen3.8-max':info['inspection_mode']='preloaded_no_tools'
    return info


def build_supervisor(session, model='astra', **kwargs):
    version = kwargs.pop('version', None)
    if version == 11:
        from .chunk_supervisor_v11 import ChunkCodexSupervisor, ChunkApiSupervisor
        if model == 'astra':
            return ChunkCodexSupervisor(session, **kwargs)
        return ChunkApiSupervisor(session, model=model, **kwargs)
    if version == 10:
        from .chunk_supervisor import ChunkCodexSupervisor, ChunkApiSupervisor
        if model == 'astra':
            return ChunkCodexSupervisor(session, **kwargs)
        return ChunkApiSupervisor(session, model=model, **kwargs)
    if model=='astra':
        from .supervisor import CodexSupervisor
        return CodexSupervisor(session,**kwargs)
    from .api_supervisor import QuickRouterSupervisor
    return QuickRouterSupervisor(session,model=model,**kwargs)
