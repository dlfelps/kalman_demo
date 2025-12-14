from inventory_simulator import SimulatorConfig, Simulator

config = SimulatorConfig(
    num_shelves=20,
    total_items=300,
    shelf_0_mode='leak_then_trap',
    trap_start_step=150
)

sim = Simulator(config, seed=42)
sim.step()
gt = sim.get_state()

print(f'Total items in simulation: {gt["quantity"].sum()}')
print(f'Items on shelves 1-19: {gt[gt["shelf_id"] != 0]["quantity"].sum()}')
print(f'Items on shelf 0: {gt[gt["shelf_id"] == 0]["quantity"].sum()}')
